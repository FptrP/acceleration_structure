#ifndef SAMPLERS_HPP_INCLUDED
#define SAMPLERS_HPP_INCLUDED

#include "driver.hpp"

#include <algorithm>
#include <unordered_map>
#include <functional>

namespace gpu {

  struct SamplerHashFunc {
    std::size_t operator()(const VkSamplerCreateInfo &info) const;
  
  private:
  };

  struct SamplerEqualFunc : std::binary_function<VkSamplerCreateInfo, VkSamplerCreateInfo, bool> {
    bool operator()(const VkSamplerCreateInfo &l, const VkSamplerCreateInfo &r) const;
  };

  struct SamplerPool {
    SamplerPool() {}
    SamplerPool(SamplerPool &&pool) :  samplers {std::move(pool.samplers)} { }
    ~SamplerPool();

    VkSampler get_sampler(const VkSamplerCreateInfo &info);
    const SamplerPool &operator=(SamplerPool &&pool);
  private:
    SamplerPool(const SamplerPool &) = delete;
    SamplerPool &operator=(const SamplerPool &) = delete;

    std::unordered_map<VkSamplerCreateInfo, VkSampler, SamplerHashFunc, SamplerEqualFunc> samplers;
  };

  constexpr VkSamplerCreateInfo DEFAULT_SAMPLER {
    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .magFilter = VK_FILTER_LINEAR,
    .minFilter = VK_FILTER_LINEAR,
    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .mipLodBias = 0.f,
    .anisotropyEnable = VK_FALSE,
    .maxAnisotropy = 0.f,
    .compareEnable = VK_FALSE,
    .compareOp = VK_COMPARE_OP_ALWAYS,
    .minLod = 0.f,
    .maxLod = 10.f,
    .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    .unnormalizedCoordinates = VK_FALSE
  };

  struct Sampler {
    Sampler(VkSamplerCreateInfo info) {
      info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      info.pNext = nullptr;
      VKCHECK(vkCreateSampler(internal::app_vk_device(), &info, nullptr, &handle));
    }

    ~Sampler() {
      if (handle)
        vkDestroySampler(internal::app_vk_device(), handle, nullptr);
    }

    VkSampler api_sampler() const { return handle; }

    Sampler(Sampler &&s) : handle {s.handle} { s.handle = nullptr; }

    const Sampler &operator=(Sampler &&s) {
      std::swap(handle, s.handle);
      return *this;
    }

  private:
    VkSampler handle {nullptr};

    Sampler(const Sampler &) = delete;
    const Sampler &operator=(const Sampler &) = delete;
  };

}

#endif