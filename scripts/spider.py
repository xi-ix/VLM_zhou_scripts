import os
from icrawler.builtin import BingImageCrawler, BaiduImageCrawler

# ================= 配置区 =================
# 每个关键词爬取的最大数量（建议先少设置一点跑通，比如 100）
MAX_IMAGES_PER_KEYWORD = 200

# 数据保存的主目录
BASE_DIR = 'dataset_raw'

# 关键词矩阵 (建议中英文结合，覆盖不同场景)
# 格式: (关键词, 搜索引擎)
SEARCH_TASKS = [
    # 电动车/摩托车类
    ("电动车 没戴头盔 抓拍", "baidu"),
    ("交警 查处 电动车 违法", "baidu"),
    ("motorcycle without helmet CCTV", "bing"),
    ("电动车 载人 违章", "baidu"),
    
    # 行人类
    ("行人 翻越护栏 抓拍", "baidu"),
    ("pedestrian jaywalking dashcam", "bing"),
    ("行人 闯红灯 监控", "baidu"),
    ("pedestrian crossing barrier", "bing")
]
# =========================================

def run_crawler():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    for keyword, engine in SEARCH_TASKS:
        print(f"\n🚀 开始抓取: [{keyword}] (引擎: {engine})")
        
        # 为每个关键词创建独立的子目录
        save_dir = os.path.join(BASE_DIR, keyword.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            if engine == 'bing':
                crawler = BingImageCrawler(
                    storage={'root_dir': save_dir},
                    downloader_threads=4 # 开启4线程加速
                )
            elif engine == 'baidu':
                crawler = BaiduImageCrawler(
                    storage={'root_dir': save_dir},
                    downloader_threads=4
                )
            else:
                continue
            
            # 启动抓取
            crawler.crawl(keyword=keyword, max_num=MAX_IMAGES_PER_KEYWORD)
            
        except Exception as e:
            print(f"❌ 抓取 [{keyword}] 时出错: {e}")

    print("\n✅ 所有抓取任务完成！图片已保存在", BASE_DIR)

if __name__ == '__main__':
    run_crawler()