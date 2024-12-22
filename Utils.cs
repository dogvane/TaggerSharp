using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace TaggerSharp
{
    public static class Utils
    {
        public static bool IsImage(this string fileName)
        {
            // 获取文件扩展名
            string extension = System.IO.Path.GetExtension(fileName).ToLower();
            // 筛选出指定格式的图像文件
            return extension == ".jpg" || extension == ".png" || extension == ".bmp" || extension == ".webp";
        }


        /// <summary>
        /// 调整图像大小并填充以适应目标宽度和高度，同时保持图像的纵横比。
        /// </summary>
        /// <param name="image">输入的图像张量。</param>
        /// <param name="targetWidth">目标宽度。</param>
        /// <param name="targetHeight">目标高度。</param>
        /// <returns>调整大小并填充后的图像张量。</returns>
        public static Tensor Letterbox(this Tensor image, int targetWidth, int targetHeight)
        {
            // 获取原始图像的宽度和高度
            int originalWidth = (int)image.shape[2];
            int originalHeight = (int)image.shape[1];

            // 计算缩放比例，以保持图像的纵横比
            float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

            // 计算缩放后的宽度和高度
            int scaledWidth = (int)(originalWidth * scale);
            int scaledHeight = (int)(originalHeight * scale);

            // 计算填充的左边和顶部的大小
            int padLeft = (targetWidth - scaledWidth) / 2;
            int padTop = (targetHeight - scaledHeight) / 2;

            // 调整图像大小
            Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);

            // 创建一个填充后的图像张量，并将缩放后的图像复制到填充后的图像张量中
            Tensor paddedImage = torch.ones([3, targetHeight, targetWidth], image.dtype, image.device);
            paddedImage[TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

            return paddedImage;
        }
    }
}
