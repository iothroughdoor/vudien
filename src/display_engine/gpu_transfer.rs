use std::cmp::max;
use std::ptr::copy_nonoverlapping as memcpy;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::PhysicalDeviceMemoryProperties;
use cgmath::{vec2, vec3};

use super::graphics_pipeline::Vertex;
use super::memory_management;
use super::device_queue::{DeviceQueue, DeviceQueueError};
use super::texture::{Texture, TextureDescription, TextureColorFormat};

pub struct TextureUploader {
    pub texture_description: TextureDescription,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub staging_buffer: vk::Buffer,
    pub staging_buffer_device_memory: vk::DeviceMemory,
}

pub enum TextureUploaderError {
    UnknownError,
    InitializationError,
    MemoryMappingError,
    UnsupportedTexture,
    CommandRecordingError,
}

impl TextureUploader {
    pub fn new(
        texture_description: &TextureDescription,
        logical_device: &Device, 
        physical_device_memory_properties: &PhysicalDeviceMemoryProperties)
    -> Result<Self, TextureUploaderError> {
        let vk_format = match texture_description.format {
            TextureColorFormat::GrayScale8Bit => vk::Format::R8G8B8A8_UNORM,
        };

        let (texture_image, texture_image_memory) = memory_management::create_image(
            logical_device,
            texture_description.width as u32,
            texture_description.height as u32,
            vk_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            physical_device_memory_properties
        ).map_err(|_| TextureUploaderError::InitializationError)?;

        let texture_image_view = memory_management::create_image_view(
            logical_device,
            texture_image,
            vk_format
        ).map_err(|_| TextureUploaderError::InitializationError)?;

        let (staging_buffer, staging_buffer_device_memory) = memory_management::create_buffer(
            logical_device,
            texture_description.size() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            physical_device_memory_properties
        ).map_err(|_| TextureUploaderError::InitializationError)?;

        Ok(TextureUploader {
            texture_description: texture_description.clone(),
            texture_image,
            texture_image_memory,
            texture_image_view,
            staging_buffer,
            staging_buffer_device_memory,
        })
    }

    pub fn destroy(&self, logical_device: &Device) {
        unsafe {
            logical_device.destroy_buffer(self.staging_buffer, None);
            logical_device.free_memory(self.staging_buffer_device_memory, None);
            logical_device.destroy_image_view(self.texture_image_view, None);
            logical_device.destroy_image(self.texture_image, None);
            logical_device.free_memory(self.texture_image_memory, None);
        }
    }

    pub fn upload(&self, 
              texture: Texture, 
              logical_device: &Device, 
              graphics_queue: &DeviceQueue) 
    -> Result<(), TextureUploaderError> {
        if self.texture_description != texture.description {
            return Err(TextureUploaderError::UnsupportedTexture);
        }

        let host_mapped_memory = unsafe {
            logical_device.map_memory(self.staging_buffer_device_memory,
                                      0, 
                                      texture.size() as vk::DeviceSize,
                                      vk::MemoryMapFlags::empty())
                          .map_err(|_| TextureUploaderError::MemoryMappingError)?
                                
        };
        unsafe {
            memcpy(texture.texture_bytes.as_ptr(),
                   host_mapped_memory.cast(),
                   texture.size());
            logical_device.unmap_memory(self.staging_buffer_device_memory);
        }

        self.transition_image_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            logical_device,
            graphics_queue
        )?;

        self.copy_buffer_to_image(self.texture_description.width as u32,
                                  self.texture_description.height as u32,
                                  logical_device,
                                  graphics_queue)?;

        self.transition_image_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            logical_device,
            graphics_queue
        )?;

        Ok(())
    }

    fn pixel_size(pixel_format: vk::Format) -> Result<usize, TextureUploaderError> {
        match pixel_format {
            vk::Format::R8G8B8A8_UNORM => Ok(4 as usize),
            vk::Format::R8G8B8_UNORM => Ok(3 as usize),
            vk::Format::R16G16B16A16_SFLOAT => Ok(8 as usize),
            _ => Err(TextureUploaderError::UnsupportedTexture)
        }
    }

    fn texture_size(width: usize, height: usize, pixel_format: vk::Format) 
    -> Result<usize, TextureUploaderError> {
        let pixel_size = Self::pixel_size(pixel_format)?;
        Ok(width * height * pixel_size)
    }

    fn transition_image_layout(
        &self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        logical_device: &Device,
        graphics_queue: &DeviceQueue,
    ) -> Result<(), TextureUploaderError> {
        let image = self.texture_image;
        let command_buffer = graphics_queue
            .begin_single_time_commands(logical_device)
            .map_err(|e|
                match e {
                    DeviceQueueError::InitializationError => TextureUploaderError::InitializationError,
                    _ => TextureUploaderError::InitializationError,
                }
            )?;
        let (
            src_access_mask,
            dst_access_mask,
            src_stage_mask,
            dst_stage_mask,
        ) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => {panic!("Not reachable")}
        };

        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        unsafe {
            logical_device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[barrier],
            );
        }

        graphics_queue
            .end_single_time_commands(logical_device, command_buffer)
            .map_err(|_| TextureUploaderError::CommandRecordingError)?;

        Ok(())
    }

    fn copy_buffer_to_image(
        &self,
        width: u32,
        height: u32,
        logical_device: &Device,
        graphics_queue: &DeviceQueue,
    ) -> Result<(), TextureUploaderError> {
        let buffer = self.staging_buffer;
        let image = self.texture_image;
        let command_buffer = graphics_queue.begin_single_time_commands(logical_device)
            .map_err(|e|
                match e {
                    DeviceQueueError::CommandBufferAllocationError => TextureUploaderError::CommandRecordingError,
                    _ => TextureUploaderError::UnknownError,
                }
            )?;

        let subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width, height, depth: 1 });

        unsafe {
            logical_device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        graphics_queue
            .end_single_time_commands(logical_device, command_buffer)
            .map_err(|_| TextureUploaderError::CommandRecordingError)?;

        Ok(())
    }

}

pub struct VertexUploader {
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_device_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_device_memory: vk::DeviceMemory,
    pub staging_buffer: vk::Buffer,
    pub staging_buffer_device_memory: vk::DeviceMemory,
}

pub enum VertexUploaderError {
    UnknownError,
    MemoryError,
    MemoryMappingError,
    BufferCreationError,
}

pub static VERTICES: [Vertex; 4] = [
    Vertex{position: vec2(-0.5, -0.5), color: vec3(u8::MAX, 0, 0)            , _padding: 0, tex_coord: vec2(0.0, 0.0)},
    Vertex{position: vec2(0.5, -0.5) , color: vec3(0, u8::MAX, 0)            , _padding: 0, tex_coord: vec2(1.0, 0.0)},
    Vertex{position: vec2(0.5, 0.5)  , color: vec3(0, 0, u8::MAX)            , _padding: 0, tex_coord: vec2(1.0, 1.0)},
    Vertex{position: vec2(-0.5, 0.5) , color: vec3(u8::MAX, u8::MAX, u8::MAX), _padding: 0, tex_coord: vec2(0.0, 1.0)},
];

pub const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

impl VertexUploader {
    pub fn new(
        logical_device: &Device,
        physical_device_memory_properties: &PhysicalDeviceMemoryProperties
    ) -> Result<Self, VertexUploaderError> {
        let indices_size = std::mem::size_of::<u16>() * INDICES.len();
        let vertices_size = std::mem::size_of::<Vertex>() * VERTICES.len();

        let (vertex_buffer, vertex_buffer_device_memory) = memory_management::create_buffer(
            logical_device,
            vertices_size as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            physical_device_memory_properties
        ).map_err(|_| VertexUploaderError::BufferCreationError)?;

        let (index_buffer, index_buffer_device_memory) = memory_management::create_buffer(
            logical_device,
            indices_size as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            physical_device_memory_properties
        ).map_err(|_| VertexUploaderError::MemoryError)?;

        let (staging_buffer, staging_buffer_device_memory) = memory_management::create_buffer(
            logical_device,
            max(indices_size, vertices_size) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            physical_device_memory_properties
        ).map_err(|_| VertexUploaderError::MemoryError)?;

        Ok(VertexUploader {
            vertex_buffer,
            vertex_buffer_device_memory,
            index_buffer,
            index_buffer_device_memory,
            staging_buffer,
            staging_buffer_device_memory,
        })
    }

    pub fn destroy(&self, logical_device: &Device) {
        unsafe {
            logical_device.destroy_buffer(self.staging_buffer, None);
            logical_device.free_memory(self.staging_buffer_device_memory, None);
            logical_device.destroy_buffer(self.index_buffer, None);
            logical_device.free_memory(self.index_buffer_device_memory, None);
            logical_device.destroy_buffer(self.vertex_buffer, None);
            logical_device.free_memory(self.vertex_buffer_device_memory, None);
        }
    }

    pub fn upload(
        &self,
        logical_device: &Device,
        graphics_queue: &DeviceQueue
    ) -> Result<(), VertexUploaderError> {
        let vertices_size = std::mem::size_of::<Vertex>() * VERTICES.len();
        let indices_size = std::mem::size_of::<u16>() * INDICES.len();

        let host_mapped_memory = unsafe {
            logical_device.map_memory(
                self.staging_buffer_device_memory,
                0,
                max(vertices_size, indices_size) as vk::DeviceSize,
                vk::MemoryMapFlags::empty()
            ).map_err(|_| VertexUploaderError::MemoryMappingError)?
        };

        unsafe {
            memcpy(
                VERTICES.as_ptr() as *const u8,
                host_mapped_memory.cast(),
                vertices_size
            );
            logical_device.unmap_memory(self.staging_buffer_device_memory);
        }

        memory_management::copy_buffer(
            logical_device,
            self.staging_buffer,
            self.vertex_buffer,
            vertices_size as vk::DeviceSize,
            graphics_queue
        ).map_err(|_| VertexUploaderError::UnknownError)?;

        let host_mapped_memory = unsafe {
            logical_device.map_memory(
                self.staging_buffer_device_memory,
                0,
                indices_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty()
            ).map_err(|_| VertexUploaderError::MemoryMappingError)?
        };

        unsafe {
            memcpy(
                INDICES.as_ptr() as *const u8,
                host_mapped_memory.cast(),
                indices_size
            );
            logical_device.unmap_memory(self.staging_buffer_device_memory);
        }

        memory_management::copy_buffer(
            logical_device,
            self.staging_buffer,
            self.index_buffer,
            indices_size as vk::DeviceSize,
            graphics_queue
        ).map_err(|_| VertexUploaderError::UnknownError)?;

        Ok(())
    }
}