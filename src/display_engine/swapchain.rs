use super::COLOR_FORMAT;

use super::memory_management;
use crate::display_engine::DisplayEngineError;
use crate::display_engine::DeviceQueue;
use crate::display_engine::swapchain;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;
use vulkanalia::vk::PFN_vkFlushMappedMemoryRanges;

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}
pub enum SwapchainError {
    CreationError,
    FormatNotSupported,
    PresentModeNotSupported,
    FrameBufferCreationError,
    AcquireNextImageError,
    WaitForFenceError,
    SupportDetailsCouldNotBeQueried,
}


impl SwapchainSupportDetails {
    pub fn new(instance: &Instance, 
               physical_device: vk::PhysicalDevice, 
               surface: vk::SurfaceKHR) 
    -> Result<SwapchainSupportDetails, SwapchainError> {
        unsafe {
            let capabilities = instance.get_physical_device_surface_capabilities_khr(physical_device, surface)
                .map_err(|_| SwapchainError::SupportDetailsCouldNotBeQueried)?;
            let formats = instance.get_physical_device_surface_formats_khr(physical_device, surface)
                .map_err(|_| SwapchainError::SupportDetailsCouldNotBeQueried)?;
            let present_modes = instance.get_physical_device_surface_present_modes_khr(physical_device, surface)
                .map_err(|_| SwapchainError::SupportDetailsCouldNotBeQueried)?;
            Ok(SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    pub fn valid(&self) -> bool {
        self.formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == COLOR_FORMAT && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            }).is_some() &&
        !self.present_modes.is_empty()
    }
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    pub images_in_flight: Vec<vk::Fence>,
    image_count: u32,
    image_format: vk::Format,
}

impl Swapchain {
    pub fn new(
        logical_device: &Device,
        surface: vk::SurfaceKHR,
        swapchain_support: &SwapchainSupportDetails,
        graphics_queue: &DeviceQueue,
        present_queue: &DeviceQueue,
        image_count: u32,
        surface_format: vk::SurfaceFormatKHR,
        present_mode: vk::PresentModeKHR, 
        extent: vk::Extent2D,
    ) -> Result<Self, SwapchainError> {
        let mut queue_indices = vec![];
        let swapchain_sharing_mode = if graphics_queue.family_index != present_queue.family_index {
            queue_indices.push(graphics_queue.family_index);
            queue_indices.push(present_queue.family_index);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(swapchain_sharing_mode)
            .queue_family_indices(&queue_indices)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());
    
        let swapchain = unsafe {
            logical_device.create_swapchain_khr(&swapchain_create_info, None)
                          .map_err(|_| SwapchainError::CreationError)?
        };

        let images = unsafe {
            logical_device.get_swapchain_images_khr(swapchain)
                .map_err(|e| { eprintln!("{:?}", e); SwapchainError::CreationError })?
        };

        let images_in_flight = images
            .iter()
            .map(|_| vk::Fence::null())
            .collect::<Vec<vk::Fence>>();

        let image_views = images
            .iter()
            .map(|&image| {
                memory_management::create_image_view(
                    logical_device,
                    image,
                    surface_format.format,
                )
                .map_err(|_| SwapchainError::CreationError)
            })
            .collect::<Result<Vec<_>, SwapchainError>>()?;

        return Ok(Swapchain {
            swapchain,
            extent,
            images,
            image_views,
            framebuffers: vec![],
            images_in_flight,
            image_count,
            image_format: surface_format.format,
        });
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn framebuffer(&self, index: usize) -> vk::Framebuffer {
        self.framebuffers[index]
    }

    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    pub fn image_format(&self) -> vk::Format {
        self.image_format
    }

    pub fn acquire_next_image(&mut self, 
                              logical_device: &Device, 
                              image_available_sem: &vk::Semaphore, 
                              in_flight_fence: vk::Fence) 
    -> Result<usize, SwapchainError> {
        let image_index = unsafe {
            logical_device
            .acquire_next_image_khr(
                self.swapchain,
                u64::MAX,
                *image_available_sem,
                vk::Fence::null(),
            )
            .map_err(|_| SwapchainError::AcquireNextImageError)?
            .0 as usize
        };

        if !self.images_in_flight[image_index].is_null() {
            unsafe {
                logical_device
                    .wait_for_fences(&[self.images_in_flight[image_index]], true, u64::MAX)
                    .map_err(|_| SwapchainError::WaitForFenceError)?;
            }
        }

        self.images_in_flight[image_index] = in_flight_fence;

        Ok(image_index) 
    }

    pub fn create_framebuffers(&mut self, 
                               logical_device: &Device, 
                               render_pass: vk::RenderPass) 
    -> Result<(), SwapchainError> {
        let framebuffers = self.image_views
        .iter()
        .map(|i| {
            let attachments = &[*i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(attachments)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);
            unsafe { 
                logical_device.create_framebuffer(&create_info, None)
                    .map_err(|_| SwapchainError::FrameBufferCreationError)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
        self.framebuffers = framebuffers;
        Ok(())
    }

    pub fn destroy(&self, logical_device: &Device) {
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|f| logical_device.destroy_framebuffer(*f, None));
            for &image_view in &self.image_views {
                logical_device.destroy_image_view(image_view, None);
            }
            logical_device.destroy_swapchain_khr(self.swapchain, None);
        }
    }
}
