using Google.Protobuf.WellKnownTypes;
using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace TaggerSharp
{
    public class ZipImageDataSet : WDImageDataSet
    {
        private List<ZipArchiveEntry> imageFiles = new List<ZipArchiveEntry>();
        private Device device;
        private ScalarType scalarType;
        private int size;
        private int cacheSize;
        private Dictionary<int, Tensor> cacheImages = new Dictionary<int, Tensor>();
        private int currentIndex = 0;
        private bool usePreload = false;
        ZipArchive zipArchive;

        protected override void Dispose(bool disposing)
        {
            imageFiles.Clear();
            cacheImages.Clear();
            zipArchive.Dispose();

            base.Dispose(disposing);
        }

        /// <summary>
        /// 初始化 ImageDataSet 类的实例，根据指定的参数加载图像文件并进行预处理。
        /// </summary>
        /// <param name="zipFileName">包含图片的zip文件名。</param>
        /// <param name="size">调整图像大小的目标尺寸，默认为 448。</param>
        /// <param name="cacheSize">缓存大小，默认为 800。</param>
        /// <param name="usePreload">是否使用预加载，默认为 false。</param>
        /// <param name="deviceType">设备类型，默认为 CUDA。</param>
        /// <param name="scalarType">张量的数据类型，默认为 Float16。</param>
        public ZipImageDataSet(string zipFileName, int size = 448, int cacheSize = 800, bool usePreload = false, DeviceType deviceType = DeviceType.CUDA, ScalarType scalarType = ScalarType.Float16)
        {
            // 打开一个zip文件
            zipArchive  = ZipFile.OpenRead(zipFileName);

            foreach (ZipArchiveEntry entry in zipArchive.Entries)
            {
                if (entry.Name.IsImage())
                {
                    imageFiles.Add(entry);
                }
            }

            // 调用另一个构造函数来初始化
            Init(size, cacheSize, usePreload, deviceType, scalarType);
        }


        private void Init(int size, int cacheSize, bool usePreload, DeviceType deviceType, ScalarType scalarType)
        {
            // 设置是否使用预加载
            this.usePreload = usePreload;

            // 设置默认的图像读取器
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

            // 初始化设备和数据类型
            this.device = new Device(deviceType);
            this.scalarType = scalarType;

            // 设置图像大小和缓存大小
            this.size = size;
            this.cacheSize = cacheSize;

            if (usePreload)
            {
                // 如果使用预加载，启动一个后台任务来预加载图像
                Task.Run(() =>
                {
                    while (true)
                    {
                        PreLoadImages();
                    }
                });

                // 等待缓存加载到一半或全部图像加载完成
                while (cacheImages.Count < cacheSize / 2 && cacheImages.Count < Count)
                {
                    Thread.Sleep(10);
                }
            }
        }

        /// <summary>
        /// 预加载图像到缓存中，以提高加载效率。
        /// </summary>
        private void PreLoadImages()
        {
            // 计算缓存大小的一半
            int halfSize = cacheSize / 2;

            // 遍历当前索引附近的图像索引
            for (int i = currentIndex - halfSize; i < currentIndex + halfSize; i++)
            {
                // 检查索引是否在有效范围内
                if (i > -1 && i < Count)
                {
                    // 如果缓存中不包含当前索引的图像
                    if (!cacheImages.ContainsKey(i))
                    {
                        lock (imageFiles) // 因为是一个zip文件，多线程读取同一个文件，会出问题，在不大改代码情况下，用锁最方便
                        {
                            using var imageStream = imageFiles[i].Open();
                            // 读取图像并添加到缓存中
                            Tensor Img = torchvision.io.read_image(imageStream).Letterbox(size,size);
                            cacheImages.Add(i, Img);
                        }
                    }
                }
            }

            // 找出超出缓存范围的图像索引
            List<int> list = cacheImages.Keys.Where(x => x > currentIndex + halfSize || x < currentIndex - halfSize).ToList();

            // 从缓存中移除超出范围的图像
            foreach (int i in list)
            {
                cacheImages.Remove(i);
            }

            Thread.Sleep(2);
        }

        public override long Count => imageFiles.Count;

        public override string GetFileName(long index)
        {
            return imageFiles[(int)index].FullName;
        }

        /// <summary>
        /// 根据提供的索引获取对应的图像张量和索引。
        /// </summary>
        /// <param name="index">图像的索引。</param>
        /// <returns>包含图像张量和索引的字典。</returns>
        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            // 更新当前索引
            currentIndex = (int)index;

            // 初始化输出字典
            Dictionary<string, Tensor> outputs = new Dictionary<string, Tensor>();

            // 定义一个空张量用于存储图像
            Tensor Img = torch.zeros(0);

            try
            {
                if (usePreload)
                {
                    // 如果使用预加载时，也会同步做尺寸调整，这里就不用再调整了
                    Img = cacheImages.GetValueOrDefault(currentIndex).to(scalarType, device) / 255.0f;
                }
                else
                {
                    lock (imageFiles) // 因为是一个zip文件，多线程读取同一个文件，会出问题，在不大改代码情况下，用锁最方便
                    {
                        using var imageStream = imageFiles[(int)index].Open();
                        // 先调整尺寸，再将数据放到GPU上，这样显存占用会小很多
                        // 但是仍然会存在一个问题，就是调整尺寸需要cpu参与，性能上会有损失
                        Img = torchvision.io.read_image(imageStream).Letterbox(size, size).to(scalarType, device) / 255.0f;
                    }
                }

                // 调整图像尺寸
                //Img = Letterbox(Img, size, size);

                // 将图像通道从RGB转换为BGR
                Tensor r = Img[0].unsqueeze(0);
                Tensor g = Img[1].unsqueeze(0);
                Tensor b = Img[2].unsqueeze(0);
                Img = torch.cat(new Tensor[] { b, g, r });

                // 将图像张量和索引添加到输出字典中
                outputs.Add("image", Img);
                outputs.Add("index", torch.tensor(index));
            }
            catch (InvalidDataException ex)
            {
                Console.WriteLine($"Skipping file {imageFiles[(int)index].FullName} due to error: {ex.Message}");
            }

            // 返回包含图像和索引的字典
            return outputs;
        }

    }
}
