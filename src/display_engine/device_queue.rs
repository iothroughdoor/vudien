use vulkanalia::prelude::v1_0::*;

use super::swapchain::Swapchain;
use super::graphics_pipeline::Pipeline;

pub enum DeviceQueueError {
    UnkownError,
    InitializationError,
    CommandPoolCreationError,
    CommandBufferAllocationError,
    CommandBufferBeginError,
    CommandBufferEndError,
    CommandBufferSubmissionError,
    QueueWaitError,
}

pub struct DeviceQueue {
    pub family_index: u32,
    pub handle: vk::Queue,
    pub command_pool: Option<vk::CommandPool>,
    pub command_buffers: Option<Vec<vk::CommandBuffer>>,
}

impl DeviceQueue {
    pub fn create_command_infrastructure(
        &mut self, 
        logical_device: &Device, 
        image_count: u32) 
    -> Result<(), DeviceQueueError> {
        let command_buffer_count = image_count;

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(self.family_index);
        let command_pool = unsafe {
            logical_device
                .create_command_pool(&command_pool_create_info, None)
                .map_err(|_| DeviceQueueError::CommandPoolCreationError)?
        };
        let command_buffers_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(command_buffer_count);
        let command_buffers = unsafe {
            logical_device.allocate_command_buffers(&command_buffers_alloc_info)
                .map_err(|_| DeviceQueueError::CommandBufferAllocationError)?
        };
        self.command_pool = Some(command_pool);
        self.command_buffers = Some(command_buffers);

        Ok(())
    }

    pub fn begin_single_time_commands(&self, logical_device: &Device) -> Result<vk::CommandBuffer, DeviceQueueError> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool.unwrap())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        
        let command_buffer = unsafe {
            logical_device
                .allocate_command_buffers(&command_buffer_allocate_info)  
                .map_err(|_| DeviceQueueError::CommandBufferAllocationError)?[0]
        };

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            logical_device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .map_err(|_| DeviceQueueError::CommandBufferBeginError)?;
        }

        Ok(command_buffer)
    }

    pub fn end_single_time_commands(
        &self, 
        logical_device: &Device, 
        command_buffer: vk::CommandBuffer) 
    -> Result<(), DeviceQueueError> {
        unsafe {
            logical_device
                .end_command_buffer(command_buffer)
                .map_err(|_| DeviceQueueError::CommandBufferEndError)?;
        }
        
        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder()
            .command_buffers(command_buffers);

        unsafe {
            logical_device
                .queue_submit(self.handle, &[info.build()], vk::Fence::null())
                .map_err(|_| DeviceQueueError::CommandBufferSubmissionError)?;
            logical_device
                .queue_wait_idle(self.handle)
                .map_err(|_| DeviceQueueError::QueueWaitError)?;

            logical_device.free_command_buffers(self.command_pool.unwrap(), command_buffers);
        }

        Ok(())
    }

    pub fn record_command_buffers<F>(&self,     
                                     logical_device: &Device, 
                                     swapchain: &Swapchain,
                                     render_pass: vk::RenderPass, 
                                     pipeline: &Pipeline,
                                     record_drawing: F)
        where F: Fn(&Device, vk::CommandBuffer)
    {
        for (i, &command_buffer) in self.command_buffers.as_ref().unwrap().iter().enumerate() {
            let begin_info = vk::CommandBufferBeginInfo::builder();
            unsafe {
                logical_device.begin_command_buffer(command_buffer, &begin_info).unwrap();
            }
            let render_area = vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(swapchain.extent());
            let color_clear_value = vk::ClearValue {
                    color: vk::ClearColorValue {
                        uint32: [0, 0, 0, u32::MAX],
                    },
            };
            let clear_values = &[color_clear_value];
            let info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(swapchain.framebuffer(i))
                .render_area(render_area)
                .clear_values(clear_values);
            unsafe {
                logical_device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE);
                logical_device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.vk_pipeline);
                logical_device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout,
                    0,
                    &[pipeline.descriptor_sets[i]],
                    &[],
                );
                record_drawing(logical_device, command_buffer);
                logical_device.cmd_end_render_pass(command_buffer);
                logical_device.end_command_buffer(command_buffer).unwrap();
            }
        }
    }

    pub fn destroy(&self, logical_device: &Device) {
        if let Some(command_pool) = self.command_pool {
            unsafe {
                logical_device.destroy_command_pool(command_pool, None);
            }
        }
        
    }
}
