#ifndef GPU_SWAPCHAIN_HPP_INCLUDED
#define GPU_SWAPCHAIN_HPP_INCLUDED

#include "driver.hpp"
#include "resources.hpp"

namespace gpu {

  struct Swapchain {
    Swapchain(VkExtent2D window, VkImageUsageFlags image_usage);
    ~Swapchain();

    Swapchain(Swapchain &&o) 
      : handle {o.handle}, descriptor {o.descriptor}
    {
      o.handle = nullptr;
    }

    const Swapchain &operator=(Swapchain &&o) {
      std::swap(handle, o.handle);
      std::swap(descriptor, o.descriptor);
      return *this;
    }

    uint32_t get_images_count() const;

    VkSwapchainKHR api_swapchain() const { return handle; }
    const ImageInfo &get_image_info() const { return descriptor; }
  private:
    VkSwapchainKHR handle {nullptr};
    ImageInfo descriptor {};

    Swapchain(const Swapchain&) = delete;
    const Swapchain& operator=(const Swapchain&) = delete;
  };

}

#endif