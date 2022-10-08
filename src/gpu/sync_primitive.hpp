#ifndef SYNC_PRIMIRIVE_HPP_INCLUDED
#define SYNC_PRIMIRIVE_HPP_INCLUDED

#include <algorithm>

#include "driver.hpp"
#include "resources.hpp"

namespace gpu {

  struct Fence {
    Fence(bool signaled = false) {

      VkFenceCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = signaled? VK_FENCE_CREATE_SIGNALED_BIT : 0u
      };

      VKCHECK(vkCreateFence(internal::app_vk_device(), &info, nullptr, &handle));
    }

    ~Fence() {
      if (handle) { vkDestroyFence(internal::app_vk_device(), handle, nullptr); }
    }

    Fence(Fence &&f) : handle {f.handle} {
      f.handle = nullptr;
    }

    const Fence &operator=(Fence &&o) {
      std::swap(handle, o.handle);
      return *this;
    }

    void reset() { vkResetFences(internal::app_vk_device(), 1, &handle); }
    
    operator VkFence() const { return handle; }

  private:
    VkFence handle {nullptr};

    Fence(const Fence&) = delete;
    const Fence &operator=(const Fence&) = delete;
  };

  struct Semaphore {
    Semaphore() {

      VkSemaphoreCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0
      };

      VKCHECK(vkCreateSemaphore(internal::app_vk_device(), &info, nullptr, &handle));
    }

    ~Semaphore() {
      if (handle) { vkDestroySemaphore(internal::app_vk_device(), handle, nullptr); }
    }

    Semaphore(Semaphore &&f) : handle {f.handle} {
      f.handle = nullptr;
    }

    const Semaphore &operator=(Semaphore &&o) {
      std::swap(handle, o.handle);
      return *this;
    }

    operator VkSemaphore() const { return handle; }

  private:
    VkSemaphore handle {nullptr};

    Semaphore(const Semaphore&) = delete;
    const Semaphore &operator=(const Semaphore&) = delete;
  };
}

#endif