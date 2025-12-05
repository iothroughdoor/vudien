pub mod texture;
pub mod graphics_pipeline;
mod swapchain;
mod device_queue;
mod memory_management;
mod gpu_transfer;

use std::collections::HashSet;
use std::ptr::copy_nonoverlapping as memcpy;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands};
use cgmath::{vec3, Deg};

use swapchain::SwapchainSupportDetails;
use swapchain::Swapchain;
use device_queue::{DeviceQueue, DeviceQueueError};
use graphics_pipeline::{Mat4, UniformBufferObject};
use texture::Texture;
use gpu_transfer::{TextureUploader,TextureUploaderError};

const VALIDATION_ENABLED: bool =
    cfg!(debug_assertions);

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const TEXTURE_WIDTH: u32 = 976;
const TEXTURE_HEIGHT: u32 = 976;
const PIXEL_DEPTH_IN_BYTES: u32 = 1;
const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Debug)]
pub enum DisplayEngineError {
    UnknownError,
    InitializationError,
    NoGraphicsDeviceFound,
    DeviceExtensionEnumerationError,
    CreateLogicalDeviceError,
    NotSupportedDeviceExtensionsError,
    MemoryError,
    VulkanError(vk::Result),
    UnsupportedValidationLayers,
    ResetFencesError,
    WaitForFenceError,
    FormatNotSupported,
    PresentModeNotSupported,
    UnsupportedTextureFormat,
    CommandError,
}

pub struct DisplayEngine {
    vk_instance: Instance,
    surface: vk::SurfaceKHR,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    logical_device: Device,
    graphics_queue: DeviceQueue,
    present_queue: DeviceQueue,
    swapchain: Swapchain,
    pipeline: graphics_pipeline::Pipeline,
    render_pass: vk::RenderPass,
    image_available_semaphores: Vec<vk::Semaphore>, // one for each frame in flight
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    vertex_buffer: Option<vk::Buffer>,
    vertex_device_memory: Option<vk::DeviceMemory>,
    index_buffer: Option<vk::Buffer>,
    index_device_memory: Option<vk::DeviceMemory>,
    texture_uploader: TextureUploader,
    texture_sampler: vk::Sampler,
}


impl DisplayEngine {
    pub fn new(window: &winit::window::Window) -> Result<Self, DisplayEngineError> {
        let vk_instance = Self::create_vk_instance(&window)?;

        let surface = unsafe { 
            vulkanalia::window::create_surface(&vk_instance, &window, &window)
                               .map_err(|_| DisplayEngineError::InitializationError)? 
        };

        let device_extensions = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
        let (physical_device, 
             graphics_queue_family_index, 
             present_queue_family_index) = 
        Self::select_physical_device(&vk_instance, &surface, device_extensions)?;

        let logical_device = Self::create_logical_device(graphics_queue_family_index, 
                                                         present_queue_family_index, 
                                                         physical_device, 
                                                         &vk_instance, 
                                                         device_extensions)?;
        let mut graphics_queue = unsafe {
            let handle = logical_device.get_device_queue(graphics_queue_family_index, 0);
            DeviceQueue {
                family_index: graphics_queue_family_index,
                handle,
                command_pool: Option::None,
                command_buffers: Option::None
            }
        };
        let present_queue = unsafe {
            let handle = logical_device.get_device_queue(present_queue_family_index, 0);
            DeviceQueue {
                family_index: present_queue_family_index,
                handle,
                command_pool: Option::None,
                command_buffers: Option::None
            }
        };

        let swapchain_support = SwapchainSupportDetails::new(&vk_instance, physical_device, surface)
                                                        .map_err(|_| DisplayEngineError::InitializationError)?;
        let mut swapchain = Self::create_swapchain(&swapchain_support, 
                                                   &logical_device, 
                                                   surface, 
                                                   window, 
                                                   &graphics_queue, 
                                                   &present_queue)
                                 .map_err(|_| DisplayEngineError::InitializationError)?;

        graphics_queue.create_command_infrastructure(&logical_device, swapchain.image_count())
                      .map_err(|_| DisplayEngineError::InitializationError)?;


        let render_pass = Self::create_render_pass(&logical_device, swapchain.image_format())
                               .map_err(|_| DisplayEngineError::InitializationError)?;

        swapchain.create_framebuffers(&logical_device, render_pass)
                 .map_err(|_| DisplayEngineError::InitializationError)?;

        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();
        for _  in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                image_available_semaphores.push(logical_device.create_semaphore(&semaphore_create_info, None)
                    .map_err(|_| DisplayEngineError::InitializationError)?);
                render_finished_semaphores.push(logical_device.create_semaphore(&semaphore_create_info, None)
                    .map_err(|_| DisplayEngineError::InitializationError)?);
                in_flight_fences.push(logical_device.create_fence(&fence_create_info, None)
                    .map_err(|_| DisplayEngineError::InitializationError)?);
            }
        }

        let physical_device_memory_properties = unsafe {
            vk_instance.get_physical_device_memory_properties(physical_device)
        };

        let texture_uploader = TextureUploader::new(TEXTURE_WIDTH, 
                                                    TEXTURE_HEIGHT, 
                                                    COLOR_FORMAT, 
                                                    &logical_device, 
                                                    &physical_device_memory_properties)
            .map_err(|_| DisplayEngineError::InitializationError)?;

        let texture_sampler_create_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);
        let texture_sampler = unsafe {
            logical_device.create_sampler(&texture_sampler_create_info, None)
                .map_err(|_| DisplayEngineError::InitializationError)?
        };

        let pipeline = graphics_pipeline::Pipeline::new(&logical_device, 
                                                        &swapchain, 
                                                        render_pass, 
                                                        "D:\\dev\\multi-platform-medical-imaging\\data", 
                                                        &physical_device_memory_properties, 
                                                        texture_uploader.texture_image_view,
                                                        texture_sampler)
            .map_err(|_| DisplayEngineError::InitializationError)?;

        return Ok(DisplayEngine{vk_instance, 
                                surface, 
                                physical_device_memory_properties,
                                logical_device, 
                                graphics_queue, 
                                present_queue,
                                swapchain,
                                pipeline,
                                render_pass,
                                image_available_semaphores,
                                render_finished_semaphores,
                                in_flight_fences,
                                vertex_buffer: None,
                                vertex_device_memory: None,
                                index_buffer: None,
                                index_device_memory: None,
                                texture_uploader,
                                texture_sampler,
                                current_frame: 0
                                });

    }

    pub fn upload_indices(&mut self, indices: &[u16]) -> Result<(), DisplayEngineError> {
        let size = std::mem::size_of::<u16>() * indices.len();
        let (staging_buffer, staging_buffer_dev_memory) = memory_management::create_buffer(
            &self.logical_device,
            size as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.physical_device_memory_properties
        ).map_err(|_| DisplayEngineError::MemoryError)?;
        let host_mapped_memory = unsafe {
            self.logical_device.map_memory(staging_buffer_dev_memory,
                                           0, 
                                           size as vk::DeviceSize,
                                           vk::MemoryMapFlags::empty())
                                .map_err(|_| DisplayEngineError::MemoryError)?
        };
        unsafe {
            memcpy(
                indices.as_ptr(),
                host_mapped_memory.cast(),
                indices.len()
            );
            self.logical_device.unmap_memory(staging_buffer_dev_memory);
        };


        let (index_buffer, index_device_memory) = memory_management::create_buffer(
            &self.logical_device,
            size as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.physical_device_memory_properties
        ).map_err(|_| DisplayEngineError::MemoryError)?;

        self.copy_buffer(
            &self.logical_device,
            staging_buffer,
            index_buffer,
            size as vk::DeviceSize,
        )?;

        unsafe {
            self.logical_device.destroy_buffer(staging_buffer, None);
            self.logical_device.free_memory(staging_buffer_dev_memory, None);
        };

        self.index_buffer = Some(index_buffer);
        self.index_device_memory = Some(index_device_memory);

        Ok(())
    }

    pub fn upload_texture(&mut self, texture: Texture) -> Result<(), DisplayEngineError> {
        self.texture_uploader
            .upload(texture, 
                    &self.logical_device, 
                    &self.graphics_queue)
            .map_err(|e| {
                match e {
                    TextureUploaderError::UnsupportedFormat => DisplayEngineError::UnsupportedTextureFormat,
                    _ => {DisplayEngineError::UnknownError}
                }
            }) 
    }

    fn copy_buffer(
        &self,
        logical_device: &Device,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<(), DisplayEngineError> {
        unsafe {
            let command_buffer = self.graphics_queue
                .begin_single_time_commands(logical_device)
                .map_err(|e| match e {
                    DeviceQueueError::CommandBufferBeginError => DisplayEngineError::CommandError,
                    DeviceQueueError::CommandBufferAllocationError => DisplayEngineError::MemoryError,
                    _ => DisplayEngineError::UnknownError,
                })?;

            let regions = vk::BufferCopy::builder().size(size);

            logical_device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

            self.graphics_queue
                .end_single_time_commands(logical_device, command_buffer)
                .map_err(|_| DisplayEngineError::CommandError)?;
        }

        Ok(())
    }

    pub fn upload_vertices(&mut self, vertices: &[graphics_pipeline::Vertex], indices: &[u16]) -> Result<(), DisplayEngineError> {
        let (vertex_buffer, vertex_device_memory) = memory_management::create_buffer(
            &self.logical_device,
            (std::mem::size_of::<graphics_pipeline::Vertex>() * vertices.len()) as vk::DeviceSize,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.physical_device_memory_properties
        ).map_err(|_| DisplayEngineError::MemoryError)?;

        let host_mapped_memory = unsafe {
            self.logical_device.map_memory(vertex_device_memory,
                                           0, 
                                           (std::mem::size_of::<graphics_pipeline::Vertex>() * vertices.len()) as vk::DeviceSize,
                                           vk::MemoryMapFlags::empty())
                                .map_err(|_| DisplayEngineError::MemoryError)?
        };
        unsafe {
            memcpy(
                vertices.as_ptr(),
                host_mapped_memory.cast(),
                vertices.len()
            );
            self.logical_device.unmap_memory(vertex_device_memory);
        }

        self.vertex_buffer = Some(vertex_buffer);
        self.vertex_device_memory = Some(vertex_device_memory);

        self.upload_indices(indices)?;

        self.graphics_queue.record_command_buffers(&self.logical_device, 
                                                   &self.swapchain, 
                                                   self.render_pass, 
                                                   &self.pipeline, 
                                                   |logical_device: &Device, command_buffer: vk::CommandBuffer| unsafe {
            logical_device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
            logical_device.cmd_bind_index_buffer(command_buffer, self.index_buffer.unwrap(), 0, vk::IndexType::UINT16);
            logical_device.cmd_draw_indexed(command_buffer, indices.len() as u32, 1, 0, 0, 0);
        });

        Ok(())
    }

    pub fn display(&mut self) -> Result<(), DisplayEngineError> {
        unsafe {
            self.logical_device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]], 
                true, 
                u64::MAX
            )
            .map_err(|_| DisplayEngineError::WaitForFenceError)?;
        }

        let image_index = self.swapchain.acquire_next_image(&self.logical_device, 
                                                            &self.image_available_semaphores[self.current_frame], 
                                                            self.in_flight_fences[self.current_frame])
            .map_err(|_| DisplayEngineError::InitializationError)?;

        unsafe {
            if !self.swapchain.images_in_flight[image_index].is_null() {
                let fence = self.swapchain.images_in_flight[image_index];
                self.logical_device.wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|_| DisplayEngineError::WaitForFenceError)?;
            }
            self.swapchain.images_in_flight[image_index as usize] =
                self.in_flight_fences[self.current_frame];

            self.update_uniform_buffer(image_index)?;
        }

        let wait_semaphores = &[self.image_available_semaphores[self.current_frame]];
        let signal_semaphores = &[self.render_finished_semaphores[self.current_frame]];
        let wait_stages = &[vk::PipelineStageFlags::TOP_OF_PIPE];
        let command_buffers = &[self.graphics_queue.command_buffers.as_ref().unwrap()[image_index]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);
        unsafe { 
            self.logical_device.reset_fences(&[self.in_flight_fences[self.current_frame]])
                .map_err(|_| DisplayEngineError::ResetFencesError)?;
            self.logical_device
                .queue_submit(self.graphics_queue.handle, &[submit_info], self.in_flight_fences[self.current_frame])
                .map_err(|_| DisplayEngineError::InitializationError)?;
        }

        let swapchains = &[self.swapchain.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        unsafe { 
            self.logical_device
                .queue_present_khr(self.present_queue.handle, &present_info)
                .map_err(|_| DisplayEngineError::InitializationError)?;
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn update_uniform_buffer(&self, image_index: usize) -> Result<(), DisplayEngineError> {
        let buffer_memory = self.pipeline.uniform_buffers_memory[image_index];

        let model = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(0.0));
        let view = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(0.0));
        let proj = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(0.0));

        let ubo = UniformBufferObject {
            model,
            view,
            proj,
        };

        let memory = unsafe { 
            self.logical_device
            .map_memory(
                buffer_memory,
                0,
                std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|_| DisplayEngineError::MemoryError)?
        };
        unsafe {
            memcpy(&ubo, memory.cast(), 1);
            self.logical_device.unmap_memory(buffer_memory);
        }

        Ok(())
    }

    fn create_vk_instance(window: &winit::window::Window) -> Result<Instance, DisplayEngineError> {
        let loader = unsafe { 
            LibloadingLoader::new(LIBRARY).map_err(|_| DisplayEngineError::InitializationError)? 
        };
        let entry = unsafe {
            Entry::new(loader).map_err(|_| DisplayEngineError::InitializationError)?
        };

        let app_info = vk::ApplicationInfo::builder()
            .application_name(b"Display Engine\0")
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 4, 312));
        let extensions = vulkanalia::window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        let available_layers = unsafe {
            entry
            .enumerate_instance_layer_properties()
            .map_err(|_| DisplayEngineError::InitializationError)?
            .iter()
            .map(|layer| layer.layer_name)
            .collect::<HashSet<_>>()
        };
        if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
            return Err(DisplayEngineError::UnsupportedValidationLayers);
        }
        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);

        unsafe {
            let instance = entry.create_instance(&create_info, None)
                .map_err(|_| DisplayEngineError::InitializationError)?;
            Ok(instance)
        }
    }

    fn select_physical_device(vk_instance: &Instance, surface: &vk::SurfaceKHR, device_extensions: &[vk::ExtensionName]) 
    -> Result<(vk::PhysicalDevice, u32, u32), DisplayEngineError> {
        unsafe {
            let mut elligible_devices = Vec::<(vk::PhysicalDevice, u32, u32)>::new();
            for physical_device in vk_instance.enumerate_physical_devices()
                                              .map_err(|_| DisplayEngineError::NoGraphicsDeviceFound)? {
                let extensions = vk_instance
                    .enumerate_device_extension_properties(physical_device, None)
                    .map_err(|_| DisplayEngineError::DeviceExtensionEnumerationError)?
                    .iter()
                    .map(|e| e.extension_name)
                    .collect::<HashSet<_>>();
                let swapchain_support = SwapchainSupportDetails::new(vk_instance, physical_device, *surface)
                                                                .map_err(|_| DisplayEngineError::InitializationError)?;
                let features = vk_instance.get_physical_device_features(physical_device);
                if device_extensions.iter().all(|ext| extensions.contains(ext)) && swapchain_support.valid() &&
                   features.sampler_anisotropy == vk::TRUE {
                    let queue_family_properties = vk_instance.get_physical_device_queue_family_properties(physical_device);

                    let graphics_queue_families = queue_family_properties
                        .iter()
                        .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                        .map(|i| i as u32);

                    let present_queue_families = queue_family_properties
                        .iter()
                        .enumerate()
                        .map(|(i, _)| i as u32)
                        .find(|i| {
                            vk_instance.get_physical_device_surface_support_khr(physical_device, *i, *surface).unwrap_or(false)
                        });

                    if let (Some(graphics_queue_family), Some(present_queue_family)) = (graphics_queue_families, present_queue_families) {
                        elligible_devices.push((physical_device, graphics_queue_family, present_queue_family));
                    } else {
                        continue;
                    }
                } 
            }
            for (physical_device, graphics_queue, present_queue) in &elligible_devices {
                let properties = vk_instance.get_physical_device_properties(*physical_device);
                if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    return Ok((*physical_device, *graphics_queue, *present_queue));
                }
            }
            if elligible_devices.first().is_some() {
                let (physical_device, graphics_queue, present_queue) = elligible_devices.first().unwrap();
                return Ok((*physical_device, *graphics_queue, *present_queue));
            }
            return Err(DisplayEngineError::NoGraphicsDeviceFound);
        }
    }

    fn create_logical_device(graphics_queue_family_index: u32,
                             present_queue_family_index: u32,
                             physical_device: vk::PhysicalDevice,
                             vk_instance: &Instance,
                             device_extensions: &[vk::ExtensionName]) 
    -> Result<Device, DisplayEngineError> {
        let queue_priorities = &[1.0];
        let mut queue_indices = HashSet::new();
        queue_indices.insert(graphics_queue_family_index);
        queue_indices.insert(present_queue_family_index); 
        let queue_infos = queue_indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();
        let layers = vec![];
        let extensions = device_extensions.iter().map(|e| e.as_ptr()).collect::<Vec<_>>();
        let features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true);
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);
        unsafe {
            vk_instance.create_device(physical_device, &device_create_info, None)
                       .map_err(|_| DisplayEngineError::CreateLogicalDeviceError)
        }
    }

    fn create_render_pass(logical_device: &Device, swapchain_image_format: vk::Format)
    -> Result<vk::RenderPass, DisplayEngineError> {
        let attachment = vk::AttachmentDescription::builder()
            .format(swapchain_image_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let slice_attachment_ref = &[attachment_ref];
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(slice_attachment_ref);
        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        let slice_attachment = &[attachment];
        let slice_subpass = &[subpass];
        let dependencies = &[dependency];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(slice_attachment)
            .subpasses(slice_subpass)
            .dependencies(dependencies);
        unsafe { 
                logical_device.create_render_pass(&render_pass_create_info, None)
                              .map_err(|_| DisplayEngineError::InitializationError) 
        }
    }

    fn create_swapchain(swapchain_support: &SwapchainSupportDetails,
                        logical_device: &Device,
                        surface: vk::SurfaceKHR,
                        window: &winit::window::Window,
                        graphics_queue: &DeviceQueue,
                        present_queue: &DeviceQueue) 
    -> Result<Swapchain, DisplayEngineError> {
        let surface_format: vk::SurfaceFormatKHR = match swapchain_support.formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == COLOR_FORMAT && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            }) {
                Some(format) => format,
                None => return Err(DisplayEngineError::FormatNotSupported)
            };
        let swapchain_present_mode = match swapchain_support.present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::FIFO) {
                Some(mode) => mode,
                None => return Err(DisplayEngineError::PresentModeNotSupported)
            };

        let swapchain_extent = vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                swapchain_support.capabilities.min_image_extent.width,
                swapchain_support.capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                swapchain_support.capabilities.min_image_extent.height,
                swapchain_support.capabilities.max_image_extent.height,
            ))
            .build();
        let mut swapchain_image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count != 0
            && swapchain_image_count > swapchain_support.capabilities.max_image_count
        {
            swapchain_image_count = swapchain_support.capabilities.max_image_count;
        }


        Swapchain::new(&logical_device, 
                       surface, 
                       &swapchain_support,
                       &graphics_queue, 
                       &present_queue,
                       swapchain_image_count,
                       surface_format,
                       swapchain_present_mode,
                       swapchain_extent)
                  .map_err(|_| DisplayEngineError::InitializationError)
    }

    pub fn wait(&self) {
        unsafe {
            self.logical_device.device_wait_idle().unwrap();
        }
    }
}

impl Drop for DisplayEngine {
    fn drop(&mut self) {
         self.graphics_queue.destroy(&self.logical_device);
         self.pipeline.destroy(&self.logical_device);
         unsafe {
            self.render_finished_semaphores
                .iter()
                .for_each(|s| self.logical_device.destroy_semaphore(*s, None));
            self.image_available_semaphores
                .iter()
                .for_each(|s| self.logical_device.destroy_semaphore(*s, None));
            self.in_flight_fences
                .iter()
                .for_each(|f| self.logical_device.destroy_fence(*f, None));
            self.texture_uploader.destroy(&self.logical_device);
            self.logical_device.destroy_render_pass(self.render_pass, None);
        }
        self.swapchain.destroy(&self.logical_device);
        unsafe {
            if self.index_buffer != None {
                self.logical_device.destroy_buffer(self.index_buffer.unwrap(), None);
                self.logical_device.free_memory(self.index_device_memory.unwrap(), None);
            }
            if self.vertex_buffer != None {
                self.logical_device.destroy_buffer(self.vertex_buffer.unwrap(), None);
                self.logical_device.free_memory(self.vertex_device_memory.unwrap(), None);
            }
            self.logical_device.destroy_sampler(self.texture_sampler, None);
            self.vk_instance.destroy_surface_khr(self.surface, None);
            self.logical_device.destroy_device(None);
            self.vk_instance.destroy_instance(None);
        }
    }
}