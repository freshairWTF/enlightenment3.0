import os
from pathlib import Path
from typing import Literal

import sys
import pandas as pd
from numpy import nan


pd.set_option('future.no_silent_downcasting', True)


##############################################################
class DataStorage:
    """分析数据存储工具，支持 Excel 和 Parquet 格式"""

    def __init__(self, dir_: str | Path):
        """
        :param dir_: 数据存储根目录
        """
        self.dir_ = Path(dir_)
        self.dir_.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # 通用获取 excel 读取器
    # --------------------------
    @staticmethod
    def _get_excel_writer(
            path: Path,
            mode: str,
            if_sheet_exists: Literal["replace", "overlay"]
    ) -> pd.ExcelWriter:
        """
        创建 ExcelWriter 实例
        :param path: 目标路径
        :param mode: 存储模式：1）a追加, 2）w覆盖
        :param if_sheet_exists: 若是追加模式：1）合并后覆盖；2）不做处理追加
        """
        if mode == "w":
            return pd.ExcelWriter(
                path,
                mode=mode,
                engine="openpyxl"
            )
        else:
            return pd.ExcelWriter(
                path,
                mode=mode,
                engine="openpyxl",
                if_sheet_exists=if_sheet_exists
            )

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def write_df_to_excel(
            self,
            df: pd.DataFrame,
            file_name: str,
            sheet_name: str | None = None,
            mode: Literal["a", "w"] = "w",
            index: bool = True,
            if_sheet_exists: Literal["replace", "overlay"] = "replace",
            merge_original_data: bool = True,
            subset: list[str] | None = None,
            keep: Literal["first", "last", False] = "first",
            sort_by: list[str] | None = None,
    ) -> None:
        """
        写入单个 DataFrame 到 Excel
        :param df: 需要存储的数据
        :param file_name: 目标文件名（不含后缀）
        :param mode: 存储模式：1）a追加, 2）w覆盖
        :param if_sheet_exists: 若是追加模式：1）合并后覆盖；2）不做处理追加
        :param merge_original_data: 若是追加模式（1），是否合并原数据
        :param sheet_name: 工作表名
        :param index: 是否存储索引
        :param subset: 去重列表：1）检查指定列；2）检查所有列 -- 不检查索引
        :param keep: 是否存储索引
        :param sort_by: 排序列表
        """
        # 处理路径
        path = self.dir_ / f"{file_name}.xlsx"

        # 追加模式，且直接覆盖原数据时，合并原有数据
        merged_df = df
        if path.exists():
            if mode == "a" and if_sheet_exists == "replace" and merge_original_data:
                try:
                    existing_df = pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
                    if not existing_df.empty:
                        merged_df = pd.concat([existing_df, df])
                except (FileNotFoundError, ValueError):
                    # 文件或 Sheet 不存在时，直接使用新数据
                    pass
        else:
            # 文件不存在时，自动切换为覆盖模式
            mode = "w"

        # 去重
        merged_df = (
            merged_df.drop_duplicates(subset=subset, keep=keep) if subset
            else merged_df.drop_duplicates(keep=keep)
        )
        # 排序
        merged_df = merged_df.sort_values(by=sort_by) if sort_by else merged_df.sort_index()

        # 存储
        try:
            with self._get_excel_writer(path, mode, if_sheet_exists) as writer:
                merged_df.to_excel(
                    writer,
                    sheet_name="Sheet1" if sheet_name is None else sheet_name,
                    index=index
                )
        except FileNotFoundError as e:
            raise IOError(f"无法写入文件: {e}") from e

    def write_dict_to_excel(
            self,
            data: dict[str, pd.DataFrame],
            file_name: str,
            mode: Literal["a", "w"] = "w",
            index: bool = True,
            if_sheet_exists: Literal["replace", "overlay"] = "replace",
            merge_original_data: bool = True,
            subset: list[str] | None = None,
            keep: Literal["first", "last", False] = "first",
            sort_by: list[str] | None = None,
    ) -> None:
        """
        写入字典 到 Excel
        :param data: 需要存储的数据
        :param file_name: 目标文件名（不含后缀）
        :param mode: 存储模式：1）a追加, 2）w覆盖
        :param if_sheet_exists: 若是追加模式：1）合并后覆盖；2）不做处理追加
        :param merge_original_data: 若是追加模式（1），是否合并原数据
        :param index: 是否存储索引
        :param subset: 去重列表：1）检查指定列；2）检查所有列 -- 不检查索引
        :param keep: 是否存储索引
        :param sort_by: 排序列表
        """
        for key, df in data.items():
            self.write_df_to_excel(
                df, file_name, key, mode, index, if_sheet_exists, merge_original_data, subset, keep, sort_by
            )

    def write_df_to_parquet(
            self,
            df: pd.DataFrame,
            file_name: str,
            index: bool = True,
            merge_original_data: bool = True,
            merge_axis: Literal[0, 1] = 0,
            subset: list[str] | None = None,
            keep: Literal["first", "last", False] = "last",
            sort_by: list[str] | None = None,
            compression: str = "snappy"
    ) -> None:
        """
        写入单个 DataFrame 到 Parquet 文件
        :param df: 要保存的 DataFrame
        :param file_name: 目标文件名（不含后缀）
        :param merge_original_data: 若是追加模式（1），是否合并原数据
        :param merge_axis: 合并方向
        :param subset: 去重列表：1）检查指定列；2）检查所有列 -- 不检查索引
        :param keep: 是否存储索引
        :param sort_by: 排序列表
        :param index: 是否保留索引
        :param compression: 压缩算法（默认 snappy）
        """
        # 路径处理
        path = self.dir_ / f"{file_name}.parquet"
        temp_path = path.with_suffix(".parquet.tmp")
        backup_path = path.with_suffix(".parquet.bak")

        # ----------------------------
        # 数据处理
        # ----------------------------
        # 数据合并
        merged_df = df
        if path.exists() and merge_original_data:
            existing_df = pd.read_parquet(path)
            if not existing_df.empty:
                merged_df = pd.concat([existing_df, df], axis=merge_axis)

        # 去重
        if merge_axis == 0:
            merged_df = (
                merged_df.drop_duplicates(subset=subset, keep=keep) if subset
                else merged_df.drop_duplicates(keep=keep)
            )
        if merge_axis == 1:
            # 保留最后出现的列（覆盖旧列）
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep=keep)]

        # 排序
        merged_df = merged_df.sort_values(by=sort_by) if sort_by else merged_df.sort_index()

        # ----------------------------
        # 原子写入
        # ----------------------------
        try:
            merged_df.to_parquet(
                temp_path,
                index=index,
                compression=compression
            )
            # 保存备份
            if path.exists():
                if sys.platform == "win32":
                    path.replace(backup_path)
                else:
                    os.rename(path, backup_path)
            # 原子替换
            temp_path.replace(path)
        except Exception as e:
            if backup_path.exists():
                backup_path.replace(path)
            raise IOError(f"Parquet写入失败: {str(e)}") from e
        finally:
            temp_path.unlink(missing_ok=True)

        # ----------------------------
        # 后置校验
        # ----------------------------
        try:
            # 验证文件可读性
            check_df = pd.read_parquet(path)
            pd.testing.assert_frame_equal(
                merged_df.reset_index(drop=not index).replace([None], nan),
                check_df.reset_index(drop=not index).replace([None], nan),
                check_exact=False,
                atol=1e-9,
                check_dtype=False
            )
        except Exception as e:
            path.unlink(missing_ok=True)
            if backup_path.exists():
                backup_path.replace(path)
            raise RuntimeError(f"数据校验失败: {str(e)}") from e
        finally:
            # 校验成功后删除备份
            backup_path.unlink(missing_ok=True)

    def write_dict_to_parquet(
            self,
            data: dict[str, pd.DataFrame],
            index: bool = True,
            merge_original_data: bool = True,
            merge_axis: Literal[0, 1] = 0,
            subset: list[str] | None = None,
            keep: Literal["first", "last", False] = "last",
            sort_by: list[str] | None = None,
            compression: str = "snappy"
    ) -> None:
        """
        写入字典 到 Excel
        :param data: 需要存储的数据
        :param merge_original_data: 若是追加模式（1），是否合并原数据
        :param merge_axis: 合并方向
        :param subset: 去重列表：1）检查指定列；2）检查所有列 -- 不检查索引
        :param keep: 是否存储索引
        :param sort_by: 排序列表
        :param index: 是否保留索引
        :param compression: 压缩算法（默认 snappy）
        """
        for key, df in data.items():
            if isinstance(df, pd.Series):
                df = pd.DataFrame({key: df})
            self.write_df_to_parquet(
                df, key, index, merge_original_data, merge_axis, subset, keep, sort_by, compression
            )
