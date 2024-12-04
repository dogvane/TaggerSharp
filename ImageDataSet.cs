using TorchSharp;
using static TorchSharp.torch;

namespace TaggerSharp
{
	internal class ImageDataSet : utils.data.Dataset
	{
		private List<string> imageFiles = new List<string>();
		private Device device;
		private ScalarType scalarType;
		private int size;
		private int cacheSize;
		private Dictionary<int, Tensor> cacheImages = new Dictionary<int, Tensor>();
		private int currentIndex = 0;
		private bool usePreload = false;


		public ImageDataSet(string rootPath, int size = 448, int cacheSize = 800, bool usePreload = false, DeviceType deviceType = DeviceType.CUDA, ScalarType scalarType = ScalarType.Float16)
		{
			this.usePreload = usePreload;
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			string[] imagesFileNames = Directory.GetFiles(rootPath, "*.*", SearchOption.AllDirectories).Where(file =>
			{
				string extension = Path.GetExtension(file).ToLower();
				return extension == ".jpg" || extension == ".png" || extension == ".bmp" || extension == ".webp";
			}).ToArray();

			foreach (string imageFileName in imagesFileNames)
			{
				imageFiles.Add(imageFileName);
			}
			this.device = new Device(deviceType);
			this.scalarType = scalarType;
			this.size = size;
			this.cacheSize = cacheSize;

			if (usePreload)
			{
				Task.Run(() =>
				{
					while (true)
					{
						PreLoadImages();
					}
				});
				while (cacheImages.Count < cacheSize / 2 && cacheImages.Count < Count)
				{
					Thread.Sleep(10);
				}
			}
		}

		private void PreLoadImages()
		{
			int halfSize = cacheSize / 2;
			for (int i = currentIndex - halfSize; i < currentIndex + halfSize; i++)
			{
				if (i > -1 && i < Count)
				{
					if (!cacheImages.ContainsKey(i))
					{
						Tensor Img = torchvision.io.read_image(imageFiles[i]);
						cacheImages.Add(i, Img);
					}
				}
			}
			List<int> list = cacheImages.Keys.Where(x => (x > currentIndex + halfSize || x < currentIndex - halfSize)).ToList();

			foreach (int i in list)
			{
				cacheImages.Remove(i);
			}
			Thread.Sleep(2);
		}

		public override long Count => imageFiles.Count;

		public string GetFileName(long index)
		{
			return imageFiles[(int)index];
		}

		public override Dictionary<string, Tensor> GetTensor(long index)
		{
			currentIndex = (int)index;
			Dictionary<string, Tensor> outputs = new Dictionary<string, Tensor>();
			Tensor Img = torch.zeros(0);
			if (usePreload)
			{
				Img = cacheImages.GetValueOrDefault(currentIndex).to(scalarType, device) / 255.0f;
			}
			else
			{
				Img = torchvision.io.read_image(imageFiles[(int)index]).to(scalarType, device) / 255.0f;
			}
			Img = Letterbox(Img, size, size);
			Tensor r = Img[0].unsqueeze(0);
			Tensor g = Img[1].unsqueeze(0);
			Tensor b = Img[2].unsqueeze(0);
			Img = torch.cat([b, g, r]);
			outputs.Add("image", Img);
			outputs.Add("index", torch.tensor(index));
			return outputs;
		}

		private Tensor Letterbox(Tensor image, int targetWidth, int targetHeight)
		{
			int originalWidth = (int)image.shape[2];
			int originalHeight = (int)image.shape[1];

			float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

			int scaledWidth = (int)(originalWidth * scale);
			int scaledHeight = (int)(originalHeight * scale);

			int padLeft = (targetWidth - scaledWidth) / 2;
			//int padRight = targetWidth - scaledWidth - padLeft;
			int padTop = (targetHeight - scaledHeight) / 2;
			//int padBottom = targetHeight - scaledHeight - padTop;

			Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);
			Tensor paddedImage = torch.ones([3, targetHeight, targetWidth], image.dtype, image.device);
			paddedImage[TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

			return paddedImage;
		}
	}
}
