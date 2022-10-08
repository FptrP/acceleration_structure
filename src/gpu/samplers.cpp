#include "samplers.hpp"

#include <iostream>
#include <cstring>
#include <cmath>

namespace gpu {

  std::size_t SamplerHashFunc::operator()(const VkSamplerCreateInfo &info) const {
    std::size_t h = 0;
    hash_combine(h, info.magFilter);
    hash_combine(h, info.minFilter);
    hash_combine(h, info.mipmapMode);
    hash_combine(h, info.addressModeU);
    hash_combine(h, info.addressModeV);
    hash_combine(h, info.addressModeW);
    hash_combine(h, info.mipLodBias);
    hash_combine(h, info.anisotropyEnable);
    hash_combine(h, info.maxAnisotropy);
    hash_combine(h, info.compareEnable);
    hash_combine(h, info.compareOp);
    hash_combine(h, info.minLod);
    hash_combine(h, info.maxLod);
    hash_combine(h, info.borderColor);
    hash_combine(h, info.unnormalizedCoordinates);
    return h;
  }

  bool SamplerEqualFunc::operator()(const VkSamplerCreateInfo &l, const VkSamplerCreateInfo &r) const {
    return 
      (l.magFilter == r.magFilter) &&
      (l.minFilter == r.minFilter) &&
      (l.mipmapMode == r.mipmapMode) &&
      (l.addressModeU == r.addressModeU) &&
      (l.addressModeV == r.addressModeV) &&
      (std::abs(l.mipLodBias - r.mipLodBias) < 1e-6) &&
      (l.anisotropyEnable == r.anisotropyEnable) &&
      (std::abs(l.maxAnisotropy - r.maxAnisotropy) < 1e-6) &&
      (l.compareEnable == r.compareEnable) &&
      (std::abs(l.minLod - r.minLod) < 1e-6) &&
      (std::abs(l.maxLod - r.maxLod) < 1e-6) &&
      (l.borderColor == r.borderColor) &&
      (l.unnormalizedCoordinates == r.unnormalizedCoordinates);
  }

  SamplerPool::~SamplerPool() {

    for (auto& [k, v] : samplers) {
      vkDestroySampler(internal::app_vk_device(), v, nullptr);
    }
  }

  const SamplerPool &SamplerPool::operator=(SamplerPool &&pool) {
    samplers = std::move(pool.samplers);
    return *this;
  }

  VkSampler SamplerPool::get_sampler(const VkSamplerCreateInfo &info) {
    //SamplerHashFunc hash_fun {};
    //std::cout << hash_fun(info) << "\n";

    auto it = samplers.find(info);
    if (it != samplers.end()) {
      return it->second;
    }

    VkSampler new_sampler {nullptr};
    VKCHECK(vkCreateSampler(internal::app_vk_device(), &info, nullptr, &new_sampler));
    samplers[info] = new_sampler;
    return new_sampler;
  }

}