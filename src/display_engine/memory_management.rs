use vulkanalia::prelude::v1_0::*;

pub enum MemoryManagementError {
    MemoryRequirementsError, 
    BufferCreationError,
    ImageCreationError,
    ImageViewCreationError,
    DeviceMemoryAllocationError,
    BindingError
}

pub fn get_memory_type_index(physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties, 
                            properties: vk::MemoryPropertyFlags,  
                            requirements: vk::MemoryRequirements) 
-> Result<u32, MemoryManagementError> {
    (0..physical_device_memory_properties.memory_type_count)
        .find(|i| {
            let is_suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = physical_device_memory_properties.memory_types[*i as usize];
            is_suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| MemoryManagementError::MemoryRequirementsError)
}

pub fn create_buffer(logical_device: &Device, 
                     size: vk::DeviceSize, 
                     usage: vk::BufferUsageFlags,
                     properties: vk::MemoryPropertyFlags,
                     physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties) 
-> Result<(vk::Buffer, vk::DeviceMemory), MemoryManagementError> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        logical_device
            .create_buffer(&buffer_info, None)
            .map_err(|_| MemoryManagementError::BufferCreationError)?
    };

    let mem_requirements = unsafe {
        logical_device.get_buffer_memory_requirements(buffer)
    };

    let mem_alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(
            get_memory_type_index(
                physical_device_memory_properties, 
                properties, 
                mem_requirements
            )?
        );

    let device_memory = unsafe {
        logical_device.allocate_memory(&mem_alloc_info, None)
            .map_err(|_| MemoryManagementError::DeviceMemoryAllocationError)?
    };

    unsafe {
        logical_device.bind_buffer_memory(buffer, device_memory, 0)
            .map_err(|_| MemoryManagementError::BindingError)?
    };

    Ok((buffer, device_memory))
}

pub fn create_image(logical_device: &Device,
                    width: u32, 
                    height: u32, 
                    pixel_format: vk::Format, 
                    tiling: vk::ImageTiling, 
                    usage: vk::ImageUsageFlags, 
                    properties: vk::MemoryPropertyFlags,
                    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties)
-> Result<(vk::Image, vk::DeviceMemory), MemoryManagementError> {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(pixel_format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1) 
        .format(pixel_format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(vk::SampleCountFlags::_1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = unsafe {
        logical_device.create_image(&image_info, None)
                      .map_err(|_| MemoryManagementError::ImageCreationError)?
    };

    let requirements = unsafe {
        logical_device.get_image_memory_requirements(image)
    };

    let device_memory = allocate_device_memory(logical_device,
                                               requirements,
                                               properties,
                                               physical_device_memory_properties)?;

    unsafe {
        logical_device.bind_image_memory(image, device_memory, 0)
            .map_err(|_| MemoryManagementError::BindingError)?
    };

    Ok((image, device_memory))
}

fn allocate_device_memory(logical_device: &Device,
                          requirements: vk::MemoryRequirements,
                          properties: vk::MemoryPropertyFlags,
                          physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties)
-> Result<vk::DeviceMemory, MemoryManagementError> {
    let mem_alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(
            get_memory_type_index(
                physical_device_memory_properties, 
                properties, 
                requirements
            )?
        );
    
    unsafe {
        logical_device.allocate_memory(&mem_alloc_info, None)
                      .map_err(|_| MemoryManagementError::DeviceMemoryAllocationError)
    }

}

pub fn create_image_view(logical_device: &Device,
                         image: vk::Image,
                         format: vk::Format)
-> Result<vk::ImageView, MemoryManagementError> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    unsafe {
        Ok(logical_device.create_image_view(&info, None)
                         .map_err(|_| MemoryManagementError::ImageViewCreationError)?)
    }
}