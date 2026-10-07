//
//  ImageAugmenter.hpp
//  neural net v1
//
//  Created by Oliver Homer on 29/09/2026.
//

#ifndef ImageAugmenter_hpp
#define ImageAugmenter_hpp

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "NeuralTypes.hpp"

class ImageAugmenter
{
public:
    void translate(const Scalar* input,
                   Scalar* output,
                   std::size_t width,
                   std::size_t height,
                   int offsetX,
                   int offsetY) const;

    void flip(const Scalar* input,
                   Scalar* output,
                   std::size_t width,
                   std::size_t height) const;
    
    
    void rotate(const Scalar* input,
                Scalar* output,
                std::size_t width,
                std::size_t height,
                Scalar theta) const;

    void rotate(const Scalar* input,
                Scalar* output,
                std::size_t width,
                std::size_t height,
                std::size_t rotationIndex) const;

    [[nodiscard]] std::size_t rotationCount() const;

private:
    static constexpr std::size_t rotationImageSide_ = 28;
    static constexpr std::size_t rotationImageSize_ = rotationImageSide_ * rotationImageSide_;
    static constexpr Scalar pi_ = 3.14159265358979323846f;
    static constexpr std::array<int, 5> rotationDegrees_ = {-5, -3, 0, 3, 5};

    struct RotationContribution
    {
        std::size_t targetIndex = 0;
        Scalar weight = 0.0f;
    };

    struct SourcePixelRotation
    {
        std::array<RotationContribution, 4> contributions;
        std::size_t count = 0;
    };

    struct RotationLookup
    {
        int degrees = 0;
        std::array<SourcePixelRotation, rotationImageSize_> sourcePixels;
    };

    static void addContribution(SourcePixelRotation& sourcePixel, int targetX, int targetY, Scalar weight);
    [[nodiscard]] static RotationLookup makeRotationLookup(int degrees);
    [[nodiscard]] static const std::array<RotationLookup, 5>& rotationLookups();
    [[nodiscard]] static std::size_t nearestRotationIndexForDegrees(int degrees);
    [[nodiscard]] std::size_t nearestRotationIndex(Scalar theta) const;
};

inline void ImageAugmenter::translate(const Scalar* input,
                                      Scalar* output,
                                      std::size_t width,
                                      std::size_t height,
                                      int offsetX,
                                      int offsetY) const
{
    if(input == nullptr || output == nullptr)
        throw std::runtime_error("ImageAugmenter::translate received null image data");

    std::fill(output, output + width * height, 0.0f);

    for(std::size_t targetY = 0; targetY < height; targetY++)
    {
        for(std::size_t targetX = 0; targetX < width; targetX++)
        {
            const int sourceX = static_cast<int>(targetX) + offsetX;
            const int sourceY = static_cast<int>(targetY) + offsetY;
            if(sourceX < 0 ||
               sourceY < 0 ||
               sourceX >= static_cast<int>(width) ||
               sourceY >= static_cast<int>(height))
            {
                continue;
            }

            output[targetY * width + targetX] =
                input[static_cast<std::size_t>(sourceY) * width + static_cast<std::size_t>(sourceX)];
        }
    }
}


inline void ImageAugmenter::flip(const Scalar* input,
                                      Scalar* output,
                                      std::size_t width,
                                      std::size_t height,
                                    std::size_t channels) const
{
    if(input == nullptr || output == nullptr)
        throw std::runtime_error("ImageAugmenter::translate received null image data");

    std::fill(output, output + width * height * channels, 0.0f);

    for(std::size_t chan = 0; chan < channels; chan++)
    for(std::size_t targetY = 0; targetY < height; targetY++)
    {
        for(std::size_t targetX = 0; targetX < width; targetX++)
        {
            const int sourceX = static_cast<int>(width-targetX-1);
            const int sourceY = static_cast<int>(targetY);
        
            output[chan * (width * height) + targetY * width + targetX] =
                input[chan * (width * height) + (static_cast<std::size_t>(sourceY) * width) + static_cast<std::size_t>(sourceX)];
        }
    }
}

inline void ImageAugmenter::rotate(const Scalar* input,
                                   Scalar* output,
                                   std::size_t width,
                                   std::size_t height,
                                   Scalar theta) const
{
    rotate(input, output, width, height, nearestRotationIndex(theta));
}

inline void ImageAugmenter::rotate(const Scalar* input,
                                   Scalar* output,
                                   std::size_t width,
                                   std::size_t height,
                                   std::size_t rotationIndex) const
{
    if(input == nullptr || output == nullptr)
        throw std::runtime_error("ImageAugmenter::rotate received null image data");

    if(width != rotationImageSide_ || height != rotationImageSide_)
        throw std::runtime_error("Precomputed rotation lookup only supports 28x28 images");

    if(rotationIndex >= rotationLookups().size())
        throw std::runtime_error("Rotation lookup index is out of range");

    std::fill(output, output + rotationImageSize_, 0.0f);
    const RotationLookup& lookup = rotationLookups()[rotationIndex];

    for(std::size_t sourceIndex = 0; sourceIndex < rotationImageSize_; sourceIndex++)
    {
        const Scalar sourceValue = input[sourceIndex];
        const SourcePixelRotation& sourcePixel = lookup.sourcePixels[sourceIndex];

        for(std::size_t i = 0; i < sourcePixel.count; i++)
        {
            const RotationContribution& contribution = sourcePixel.contributions[i];
            output[contribution.targetIndex] += sourceValue * contribution.weight;
        }
    }
}

inline std::size_t ImageAugmenter::rotationCount() const
{
    return rotationLookups().size();
}

inline void ImageAugmenter::addContribution(SourcePixelRotation& sourcePixel, int targetX, int targetY, Scalar weight)
{
    if(weight <= 0.0f ||
       targetX < 0 ||
       targetY < 0 ||
       targetX >= static_cast<int>(rotationImageSide_) ||
       targetY >= static_cast<int>(rotationImageSide_))
    {
        return;
    }

    sourcePixel.contributions[sourcePixel.count] = {
        static_cast<std::size_t>(targetY) * rotationImageSide_ + static_cast<std::size_t>(targetX),
        weight
    };
    sourcePixel.count++;
}

inline ImageAugmenter::RotationLookup ImageAugmenter::makeRotationLookup(int degrees)
{
    RotationLookup lookup;
    lookup.degrees = degrees;

    const Scalar theta = static_cast<Scalar>(degrees) * pi_ / 180.0f;
    const Scalar cosTheta = std::cos(theta);
    const Scalar sinTheta = std::sin(theta);
    const Scalar centre = static_cast<Scalar>(rotationImageSide_ - 1) / 2.0f;

    for(std::size_t sourceY = 0; sourceY < rotationImageSide_; sourceY++)
    {
        for(std::size_t sourceX = 0; sourceX < rotationImageSide_; sourceX++)
        {
            const Scalar x = static_cast<Scalar>(sourceX) - centre;
            const Scalar y = static_cast<Scalar>(sourceY) - centre;
            const Scalar targetX = cosTheta * x - sinTheta * y + centre;
            const Scalar targetY = sinTheta * x + cosTheta * y + centre;

            const int x0 = static_cast<int>(std::floor(targetX));
            const int y0 = static_cast<int>(std::floor(targetY));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const Scalar xWeight = targetX - static_cast<Scalar>(x0);
            const Scalar yWeight = targetY - static_cast<Scalar>(y0);

            SourcePixelRotation& sourcePixel = lookup.sourcePixels[sourceY * rotationImageSide_ + sourceX];
            addContribution(sourcePixel, x0, y0, (1.0f - xWeight) * (1.0f - yWeight));
            addContribution(sourcePixel, x1, y0, xWeight * (1.0f - yWeight));
            addContribution(sourcePixel, x0, y1, (1.0f - xWeight) * yWeight);
            addContribution(sourcePixel, x1, y1, xWeight * yWeight);
        }
    }

    return lookup;
}

inline const std::array<ImageAugmenter::RotationLookup, 5>& ImageAugmenter::rotationLookups()
{
    static const std::array<RotationLookup, 5> lookups = {
        makeRotationLookup(rotationDegrees_[0]),
        makeRotationLookup(rotationDegrees_[1]),
        makeRotationLookup(rotationDegrees_[2]),
        makeRotationLookup(rotationDegrees_[3]),
        makeRotationLookup(rotationDegrees_[4])
    };

    return lookups;
}

inline std::size_t ImageAugmenter::nearestRotationIndexForDegrees(int degrees)
{
    std::size_t bestIndex = 0;
    int bestDistance = std::abs(degrees - rotationDegrees_[0]);

    for(std::size_t i = 1; i < rotationDegrees_.size(); i++)
    {
        const int distance = std::abs(degrees - rotationDegrees_[i]);
        if(distance < bestDistance)
        {
            bestDistance = distance;
            bestIndex = i;
        }
    }

    return bestIndex;
}

inline std::size_t ImageAugmenter::nearestRotationIndex(Scalar theta) const
{
    const int degrees = static_cast<int>(std::round(theta * 180.0f / pi_));
    return nearestRotationIndexForDegrees(degrees);
}

#endif /* ImageAugmenter_hpp */
