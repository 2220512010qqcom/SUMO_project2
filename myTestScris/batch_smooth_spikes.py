import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def remove_spikes_with_avg(df, columns, threshold=0.4):
    """
    检测并平滑毛刺（涨跌幅超过threshold的峰值）
    
    Parameters:
    -----------
    df : pd.DataFrame
        输入数据
    columns : list
        需要处理的列名
    threshold : float
        涨跌幅阈值（默认0.4即40%）
    
    Returns:
    --------
    df_cleaned : pd.DataFrame
        处理后的数据
    spike_count : dict
        每列检测到的毛刺数量
    """
    df_cleaned = df.copy()
    spike_count = {col: 0 for col in columns}
    
    for col in columns:
        data = df_cleaned[col].values
        n = len(data)
        
        for i in range(1, n-1):  # 跳过第一个和最后一个点
            prev_val = data[i-1]
            curr_val = data[i]
            next_val = data[i+1]
            
            # 计算涨跌幅（避免除零）
            if prev_val != 0:
                change = abs(curr_val - prev_val) / prev_val
            else:
                change = float('inf') if curr_val != 0 else 0
            
            # 检查是否为峰值（高于两侧）
            is_peak = (curr_val > prev_val and curr_val > next_val)
            
            # 如果是峰值且涨跌幅超过阈值
            if is_peak and change > threshold:
                # 用相邻值的平均值替换
                new_val = (prev_val + next_val) / 2
                data[i] = new_val
                spike_count[col] += 1
        
        # 更新DataFrame
        df_cleaned[col] = data
    
    return df_cleaned, spike_count


def batch_process_csv_files(input_folder, output_folder=None, threshold=0.4, 
                           columns_to_process=['Queue_Length_Avg', 'Waiting_Time_Total', 'Emergency_Delay_Total']):
    """
    批量处理文件夹下的所有CSV文件
    
    Parameters:
    -----------
    input_folder : str
        输入文件夹路径
    output_folder : str or None
        输出文件夹路径（如果为None，则创建带"_cleaned"后缀的文件夹）
    threshold : float
        涨跌幅阈值（默认0.4即40%）
    columns_to_process : list
        需要处理的列名
    
    Returns:
    --------
    results : dict
        处理结果统计
    """
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    if not csv_files:
        print(f"在 {input_folder} 中未找到CSV文件")
        return None
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = input_folder.rstrip('/\\') + '_cleaned'
    
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 存储处理结果
    results = {
        'total_files': len(csv_files),
        'processed_files': [],
        'total_spikes': 0
    }
    
    print("=" * 80)
    print("批量CSV毛刺平滑处理")
    print("=" * 80)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"阈值: {threshold*100}%")
    print(f"处理列: {columns_to_process}")
    print(f"找到 {len(csv_files)} 个CSV文件")
    print("=" * 80)
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n正在处理: {file_name}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查必要的列是否存在
            available_columns = [col for col in columns_to_process if col in df.columns]
            missing_columns = [col for col in columns_to_process if col not in df.columns]
            
            if missing_columns:
                print(f"  警告: 缺少列 {missing_columns}，将只处理可用列: {available_columns}")
            
            if not available_columns:
                print(f"  跳过: 没有可处理的列")
                continue
            
            # 处理毛刺
            df_cleaned, spike_counts = remove_spikes_with_avg(df, available_columns, threshold)
            
            # 保存处理后的文件
            output_path = os.path.join(output_folder, file_name)
            df_cleaned.to_csv(output_path, index=False)
            
            # 统计信息
            total_spikes = sum(spike_counts.values())
            results['processed_files'].append({
                'file_name': file_name,
                'spike_counts': spike_counts,
                'total_spikes': total_spikes
            })
            results['total_spikes'] += total_spikes
            
            # 打印处理结果
            print(f"  ✓ 处理完成，共平滑 {total_spikes} 个毛刺")
            for col, count in spike_counts.items():
                if count > 0:
                    print(f"    - {col}: {count} 个")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("处理完成总结")
    print("=" * 80)
    print(f"成功处理文件数: {len(results['processed_files'])} / {results['total_files']}")
    print(f"总平滑毛刺数: {results['total_spikes']}")
    
    if results['processed_files']:
        print("\n各文件详情:")
        for item in results['processed_files']:
            print(f"  {item['file_name']}: {item['total_spikes']} 个毛刺")
    
    print(f"\n处理后的文件已保存到: {output_folder}")
    print("=" * 80)
    
    return results


def batch_process_with_subfolders(root_folder, threshold=0.4, 
                                 columns_to_process=['Queue_Length_Avg', 'Waiting_Time_Total', 'Emergency_Delay_Total']):
    """
    递归处理根文件夹下所有子文件夹中的CSV文件
    
    Parameters:
    -----------
    root_folder : str
        根文件夹路径
    threshold : float
        涨跌幅阈值
    columns_to_process : list
        需要处理的列名
    """
    # 查找所有CSV文件
    all_csv_files = glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True)
    
    if not all_csv_files:
        print(f"在 {root_folder} 中未找到CSV文件")
        return
    
    print("=" * 80)
    print(f"递归处理 {root_folder} 下所有子文件夹中的CSV文件")
    print("=" * 80)
    print(f"找到 {len(all_csv_files)} 个CSV文件")
    
    # 按文件夹分组处理
    files_by_folder = {}
    for file_path in all_csv_files:
        folder = os.path.dirname(file_path)
        if folder not in files_by_folder:
            files_by_folder[folder] = []
        files_by_folder[folder].append(file_path)
    
    # 逐文件夹处理
    for folder, files in files_by_folder.items():
        print(f"\n处理文件夹: {folder}")
        output_folder = folder + '_cleaned'
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            print(f"  处理: {file_name}")
            
            try:
                df = pd.read_csv(file_path)
                available_columns = [col for col in columns_to_process if col in df.columns]
                
                if available_columns:
                    df_cleaned, spike_counts = remove_spikes_with_avg(df, available_columns, threshold)
                    output_path = os.path.join(output_folder, file_name)
                    df_cleaned.to_csv(output_path, index=False)
                    print(f"    ✓ 完成，平滑 {sum(spike_counts.values())} 个毛刺")
                else:
                    print(f"    ⚠ 跳过：无可用列")
            except Exception as e:
                print(f"    ✗ 失败: {str(e)}")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 方式1：处理单个文件夹下的所有CSV文件
    input_folder = "../outputs/output529/10"  # 修改为你的文件夹路径，例如 "./data" 或 "C:/my_data"
    batch_process_csv_files(
        input_folder=input_folder,
        output_folder=None,  # 自动创建带"_cleaned"后缀的文件夹
        threshold=0.4,       # 40%阈值
        columns_to_process=['Queue_Length_Avg', 'Waiting_Time_Total', 'Emergency_Delay_Total']
    )
    
    # 方式2：如果需要处理包含子文件夹的整个目录树
    # root_folder = "."  # 修改为你的根文件夹路径
    # batch_process_with_subfolders(
    #     root_folder=root_folder,
    #     threshold=0.4,
    #     columns_to_process=['Queue_Length_Avg', 'Waiting_Time_Total', 'Emergency_Delay_Total']
    # )
    
    # 方式3：自定义处理逻辑（如果你的CSV文件有不同的列名）
    # custom_columns = ['Queue_Length_Avg', 'Waiting_Time_Total']  # 只处理部分列
    # batch_process_csv_files(
    #     input_folder="./your_data_folder",
    #     output_folder="./cleaned_data",
    #     threshold=0.35,  # 35%阈值
    #     columns_to_process=custom_columns
    # )