#include "swapchain.hpp"

namespace gpu {
  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
  };

  static SwapChainSupportDetails query_swapchain_info(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.present_modes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.present_modes.data());
    }

    return details;
  }

  Swapchain::Swapchain(VkExtent2D window, VkImageUsageFlags image_usage)
  {
    auto &dev = gpu::app_device();
    auto &surface = gpu::app_surface();

    auto details = query_swapchain_info(dev.api_physical_device(), surface.api_surface());

    VkSurfaceFormatKHR fmt = details.formats.at(0);
    for (auto sfmt : details.formats) {
      if (sfmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && sfmt.format == VK_FORMAT_B8G8R8A8_SRGB) {
        fmt = sfmt;
        break;
      }
    }

    VkPresentModeKHR mode = details.present_modes.at(0);

    for (auto m : details.present_modes) {
      if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
        mode = m;
      }
    }

    auto swapchain_extent = details.capabilities.currentExtent;

    if (swapchain_extent.width == UINT32_MAX) {
      swapchain_extent = window;
      swapchain_extent.width = std::clamp(swapchain_extent.width, details.capabilities.minImageExtent.width, details.capabilities.maxImageExtent.width);
      swapchain_extent.height = std::clamp(swapchain_extent.height, details.capabilities.minImageExtent.height, details.capabilities.maxImageExtent.height);
    }

    uint32_t image_count = details.capabilities.minImageCount + 1;
    if (details.capabilities.maxImageCount > 0 && image_count > details.capabilities.maxImageCount) {
      image_count = details.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = nullptr,
      .flags = 0,
      .surface = surface.api_surface(),
      .minImageCount = image_count,
      .imageFormat = fmt.format,
      .imageColorSpace = fmt.colorSpace,
      .imageExtent = swapchain_extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .preTransform = details.capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = mode,
      .clipped = VK_TRUE,
      .oldSwapchain = nullptr
    };

    VKCHECK(vkCreateSwapchainKHR(dev.api_device(), &info, nullptr, &handle));
    
    descriptor = ImageInfo {
      fmt.format,
      VK_IMAGE_ASPECT_COLOR_BIT,
      swapchain_extent.width,
      swapchain_extent.height,
    };
  }
  
  Swapchain::~Swapchain() {
    if (handle) {
      vkDestroySwapchainKHR(gpu::app_device().api_device(), handle, nullptr);
    }
  }

  uint32_t Swapchain::get_images_count() const {
    uint32_t count;
    vkGetSwapchainImagesKHR(gpu::app_device().api_device(), handle, &count, nullptr);
    return count;
  }
}