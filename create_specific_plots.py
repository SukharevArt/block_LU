import csv
import matplotlib.pyplot as plt

data = []
with open('comprehensive_benchmark_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'MatrixSize': int(row['MatrixSize']),
            'BlockSize': int(row['BlockSize']),
            'Threads': int(row['Threads']),
            'MeanTime': float(row['MeanTime'])
        })

sizes = [200, 2000, 4000]
filtered_data = [d for d in data if d['MatrixSize'] in sizes]

block_sizes = sorted(set(d['BlockSize'] for d in filtered_data))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Зависимость времени выполнения от размера блока для разных размеров матриц', fontsize=16)

for idx, size in enumerate(sizes):
    ax = axes[idx]
    size_data = [d for d in filtered_data if d['MatrixSize'] == size]
    
    block_times = {}
    for bs in block_sizes:
        bs_data = [d for d in size_data if d['BlockSize'] == bs and d['Threads'] == 1]
        if bs_data:
            block_times[bs] = sum(d['MeanTime'] for d in bs_data) / len(bs_data)
    
    sorted_blocks = sorted(block_times.keys())
    times = [block_times[bs] for bs in sorted_blocks]
    
    bars = ax.bar(range(len(sorted_blocks)), times, color='steelblue', alpha=0.7)
    
    ax.set_xlabel('Размер блока', fontsize=12)
    ax.set_ylabel('Время (мс)', fontsize=12)
    ax.set_title(f'Матрица {size}×{size}', fontsize=14)
    ax.set_xticks(range(len(sorted_blocks)))
    ax.set_xticklabels([str(bs) for bs in sorted_blocks])
    ax.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('graphs/time_vs_block_size_specific.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Зависимость времени выполнения от количества потоков для разных размеров блоков', fontsize=16)

for idx, size in enumerate(sizes):
    ax = axes[idx]
    size_data = [d for d in filtered_data if d['MatrixSize'] == size]
    
    colors = ['darkorange', 'forestgreen', 'crimson', 'blue', 'purple', 'brown']
    
    for bs_idx, bs in enumerate(block_sizes):
        bs_data = [d for d in size_data if d['BlockSize'] == bs]
        if bs_data:
            bs_sorted = sorted(bs_data, key=lambda x: x['Threads'])
            threads = [d['Threads'] for d in bs_sorted]
            times = [d['MeanTime'] for d in bs_sorted]
            
            ax.plot(threads, times, marker='o', linewidth=2, markersize=6, 
                   color=colors[bs_idx % len(colors)], label=f'Блок {bs}')
    
    ax.set_xlabel('Количество потоков', fontsize=12)
    ax.set_ylabel('Время (мс)', fontsize=12)
    ax.set_title(f'Матрица {size}×{size}', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/time_vs_threads_specific.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Ускорение при использовании нескольких потоков для разных размеров блоков', fontsize=16)

for idx, size in enumerate(sizes):
    ax = axes[idx]
    size_data = [d for d in filtered_data if d['MatrixSize'] == size]
    
    colors = ['darkorange', 'forestgreen', 'crimson', 'blue', 'purple', 'brown']
    
    for bs_idx, bs in enumerate(block_sizes):
        bs_data = [d for d in size_data if d['BlockSize'] == bs]
        if bs_data:
            bs_sorted = sorted(bs_data, key=lambda x: x['Threads'])
            threads = [d['Threads'] for d in bs_sorted]
            times = [d['MeanTime'] for d in bs_sorted]
            
            single_thread_time = times[0]
            speedup = [single_thread_time / t for t in times]
            
            ax.plot(threads, speedup, marker='o', linewidth=2, markersize=6, 
                   color=colors[bs_idx % len(colors)], label=f'Блок {bs}')
    
    all_threads = sorted(set(d['Threads'] for d in size_data))
    ax.plot(all_threads, all_threads, 'k--', linewidth=1.5, alpha=0.5, label='Идеальное')
    
    ax.set_xlabel('Количество потоков', fontsize=12)
    ax.set_ylabel('Ускорение', fontsize=12)
    ax.set_title(f'Матрица {size}×{size}', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/speedup_vs_threads_specific.png', dpi=300, bbox_inches='tight')
plt.close()

print("Графики успешно созданы:")
print("- graphs/time_vs_block_size_specific.png")
print("- graphs/time_vs_threads_specific.png") 
print("- graphs/speedup_vs_threads_specific.png")