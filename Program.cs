using System.Diagnostics;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TaggerSharp
{
	internal class Program
	{
		private static ScalarType scalarType = ScalarType.Float16;
		private static DeviceType deviceType = DeviceType.CUDA;
		private static float threshold = 0.3f;

		static void Main(string[] args)
		{
			List<string> tags = GetModelTagList(@"..\..\..\Assets\tags.dll");
			tags.RemoveRange(0, 4);
			var model = jit.load(@"..\..\..\Assets\vit_fp16.torchscript", deviceType).to(scalarType);
			ImageDataSet dataset = new ImageDataSet(@"Image\Folder\", cacheSize: 800, deviceType: deviceType, scalarType: scalarType);

			DataLoader dataLoader = new DataLoader(dataset, 64, num_worker: 512, device: CUDA, shuffle: false);
			Stopwatch sw = Stopwatch.StartNew();

			int step = 0;
			using (torch.no_grad())
			{
				foreach (var data in dataLoader)
				{
					step++;
					Console.WriteLine("Process: {0}/{1}......", step, dataLoader.Count);
					Tensor images = data["image"];
					Tensor indexs = data["index"];
					Tensor result = (Tensor)model.forward(images);
					result = result.sigmoid()[TensorIndex.Ellipsis, 4..];
					var sortedIndices = result.argsort(dim: 1, descending: true);
					var mask = result > threshold;
					Tensor countTensor = mask.sum(1, true);
					long[] counts = countTensor.data<long>().ToArray();
					for (int i = 0; i < counts.Length; i++)
					{
						StringBuilder stringBuilder = new StringBuilder();
						string name = Path.GetFileNameWithoutExtension(dataset.GetFileName(indexs[i].ToInt64()));
						long[] sortedIndex = sortedIndices[i].data<long>().ToArray();
						for (int j = 0; j < counts[i]; j++)
						{
							stringBuilder = stringBuilder.Append(tags[(int)(sortedIndex[j])] + ", ");
						}
						if (!Directory.Exists("temp"))
						{
							Directory.CreateDirectory("temp");
						}
						File.WriteAllText(Path.Combine("temp", name + ".txt"), stringBuilder.ToString());
					}
				}
			}
			Console.WriteLine("All done. Tag {0} images, using {1} seconds.", dataset.Count, sw.Elapsed.TotalSeconds);
		}
		public static List<string> GetModelTagList(string filePath)
		{
			bool IsFirst = true;
			List<string> tags = new List<string>();
			foreach (string strLine in File.ReadLines(filePath))
			{
				if (IsFirst)
				{
					IsFirst = false;
				}
				else
				{
					string[] aryLine = strLine.Split(',');
					tags.Add(aryLine[1]);
				}
			}
			return tags;


		}
	}
}
