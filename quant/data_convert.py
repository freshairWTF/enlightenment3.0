import pandas as pd

from constant.path_config import DataPATH
from data_storage import DataStorage
from data_loader import DataLoader


####################################################
class DataConvert:
    """将因子数据转换为按日期存储的格式"""

    def __init__(
            self,
            source_dir: str,
            storage_dir: str
    ):
        """
        :param source_dir: 因子源目录名称
        :param storage_dir: 存储目录名称
        """
        # 路径参数
        self.source_dir = DataPATH.QUANT_ANALYSIS_RESULT / source_dir
        self.storage_dir = DataPATH.QUANT_CONVERT_RESULT / storage_dir

        # 组件初始化
        self.storage = DataStorage(self.storage_dir)
        self._file_cache = []

    def _cache_files(
            self
    ) -> None:
        """预缓存文件路径"""
        if not self._file_cache:
            self._file_cache = [
                (f.stem, f) for f in self.source_dir.glob("*.parquet")
                if f.is_file()
            ]

    def _load_data(
            self
    ) -> dict[str, pd.DataFrame]:
        """批量加载Parquet数据"""
        self._cache_files()
        return {
            stem: DataLoader.load_parquet(
                path,
                set_date_index=True
            )
            for stem, path in self._file_cache
        }

    @staticmethod
    def _transform_data(
        data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """重构数据字典键为 日期 的格式"""
        dates = data["close"].index
        return {
            date.strftime("%Y-%m-%d"): pd.concat(
                [df.loc[date].rename(col) for col, df in data.items() if date in df.index],
                axis=1
            )
            for date in dates
        }

    def run(
            self
    ) -> None:
        """执行接口"""
        # 数据加载
        raw_data = self._load_data()

        # 数据重构
        transformed = self._transform_data(raw_data)

        # 存储
        self.storage.write_dict_to_parquet(
            transformed,
            index=True,
            merge_original_data=True,
            merge_axis=1
        )
