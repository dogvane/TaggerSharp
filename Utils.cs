using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    }
}
