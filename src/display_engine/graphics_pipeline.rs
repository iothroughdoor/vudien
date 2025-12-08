use std::path::Path;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::bytecode::Bytecode;

use crate::display_engine::swapchain::Swapchain;
use crate::display_engine::memory_management;

pub struct Pipeline {
    pub vk_pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
}

pub type Vec3 = cgmath::Vector3<u8>;
pub type Vec2 = cgmath::Vector2<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: Vec2,
    pub color: Vec3,
    pub _padding: u8,
    pub tex_coord: Vec2,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

pub enum PipelineError {
    CreationError,
    CreateVertexBufferError,
    MemoryError,
}

impl Pipeline {
    pub fn new(logical_device: &Device, 
               swapchain: &Swapchain, 
               render_pass: vk::RenderPass,
               shader_dir_path: &str,
               physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
               texture_image_view: vk::ImageView,
               texture_sampler: vk::Sampler
               )
    -> Result<Pipeline, PipelineError> {
        let binding_desc = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();

        let pos_attr_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();

        let color_attr_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R8G8B8_UNORM)
            .offset(size_of::<Vec2>() as u32)
            .build();

        let tex_coord_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec2>() + size_of::<Vec3>() + 1) as u32)
            .build();

        let binding_descs = &[binding_desc];
        let attr_descs = &[pos_attr_desc, color_attr_desc, tex_coord_desc];
        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descs)
            .vertex_attribute_descriptions(attr_descs);

        let input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(swapchain.extent().width as f32)
            .height(swapchain.extent().height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();
        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(swapchain.extent());
        let viewports = &[viewport];
        let scissors = &[scissor];  
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::_1);

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false);
        let attachments = &[color_blend_attachment_state];
        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = &[ubo_binding, sampler_binding];
        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(bindings);
        let descriptor_set_layout = unsafe {
            logical_device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .map_err(|_| PipelineError::CreationError)?
        };
        let set_layouts = &[descriptor_set_layout];

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts);
        let pipeline_layout = unsafe { 
            logical_device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|_| PipelineError::CreationError)?
        };

        let vert_path = Path::new(shader_dir_path).join("shaders\\vert.spv");
        let frag_path = Path::new(shader_dir_path).join("shaders\\frag.spv");
        let vert = std::fs::read(vert_path)
            .map_err(|_| PipelineError::CreationError)?;
        let frag = std::fs::read(frag_path)
            .map_err(|_| PipelineError::CreationError)?;
        let vert_shader_module = Self::create_shader_module(&logical_device, &vert[..])?;
        let frag_shader_module = Self::create_shader_module(&logical_device, &frag[..])?;

        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");
        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");

        let stages_info = &[vert_shader_stage_info, frag_shader_stage_info];
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages_info)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_state_info)
            .multisample_state(&multisample_state_info)
            .color_blend_state(&color_blend_state_info)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            logical_device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .map_err(|_| PipelineError::CreationError)?.0[0]
        };

        unsafe {
            logical_device.destroy_shader_module(vert_shader_module, None);
            logical_device.destroy_shader_module(frag_shader_module, None);
        } 

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(swapchain.images.len() as u32);
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(swapchain.images.len() as u32);
        let pool_sizes = &[ubo_size, sampler_size];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(swapchain.images.len() as u32);
        let descriptor_pool = unsafe {
            logical_device.create_descriptor_pool(&descriptor_pool_info, None)  
                .map_err(|_| PipelineError::CreationError)?
        };

        let layouts = vec![descriptor_set_layout; swapchain.images.len()];
        let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe {
            logical_device.allocate_descriptor_sets(&descriptor_set_alloc_info)
                .map_err(|_| PipelineError::CreationError)?
        };

        let mut uniform_buffers = Vec::new();
        let mut uniform_buffers_memory = Vec::new();
        for _ in 0..swapchain.images.len() {
            let (uniform_buffer, uniform_buffer_memory) = memory_management::create_buffer(
                &logical_device,
                size_of::<UniformBufferObject>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
                physical_device_memory_properties
            )
            .map_err(|_| PipelineError::MemoryError)?;

            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
        }

        // associate buffers with descriptor sets
        for i in 0..swapchain.images.len() {
            let descr_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64);
            let descr_buffer_infos = &[descr_buffer_info];
            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(descr_buffer_infos);

            let descr_image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture_image_view)
                .sampler(texture_sampler);
            let descr_image_infos = &[descr_image_info];
            let sampler_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(descr_image_infos);

            unsafe {
                logical_device.update_descriptor_sets(&[ubo_write, sampler_write], &[] as &[vk::CopyDescriptorSet]);
            }
        }


        Ok(Pipeline {
            vk_pipeline: pipeline,
            layout: pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffers,   
            uniform_buffers_memory,
        })
    }

    pub fn destroy(&self, logical_device: &Device) {
        unsafe {
            self.uniform_buffers
                .iter()
                .for_each(|b| logical_device.destroy_buffer(*b, None));
            self.uniform_buffers_memory
                .iter()
                .for_each(|m| logical_device.free_memory(*m, None));
            logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
            logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            logical_device.destroy_pipeline_layout(self.layout, None);
            logical_device.destroy_pipeline(self.vk_pipeline, None);
        }
    }

    fn create_shader_module(logical_device: &Device, code: &[u8]) -> Result<vk::ShaderModule, PipelineError> {
        let bytecode = Bytecode::new(code).map_err(|_| PipelineError::CreationError)?;
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(bytecode.code())
            .code_size(bytecode.code_size());
        let shader_module = unsafe {
            logical_device.create_shader_module(&shader_module_create_info, None)
                .map_err(|_| PipelineError::CreationError)?
        };
        Ok(shader_module)
    }
}