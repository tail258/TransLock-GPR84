"""主入口：归档核验，或正式训练、参数优化、完整目录筛选。"""
import argparse
from verify import verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="尚不存在的运行输出目录，包含results.csv")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="计算设备，默认cuda")
    parser.add_argument("--workflow", choices=["verify", "full"], default="verify",
                        help="verify：核验归档并导出10项清单（默认）；full：固定训练到完整目录筛选")
    parser.add_argument("--input", default="data/catalogue/structures.csv",
                        help="仅供full流程使用的固定10万条目录输入")
    args = parser.parse_args()
    if args.workflow == "full":
        from src.submission import pipeline
        pipeline(args.input, args.output, args.device)
    else:
        if args.input != "data/catalogue/structures.csv":
            parser.error("--input仅用于--workflow full；归档核验使用封存输入")
        verify(args.device, args.output)
