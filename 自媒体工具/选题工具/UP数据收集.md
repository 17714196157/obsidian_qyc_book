
# 一）B站爬取脚本，容易被限流
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站UP主视频数据收集脚本 (修复版)
适配bilibili-api-python  17.4.1最新版本
"""
import sys
import asyncio
import time
import json
from datetime import datetime
from bilibili_api import user, video, sync
import pandas as pd
def get_up_all_videos(uid, delay=1.0):
    """
    获取UP主所有视频基础信息 (修复版)
    delay: 请求间隔（秒），防止被限流
    """
    u = user.User(uid=uid)
    # 获取UP主信息
    try:
        user_info = sync(u.get_user_info())
        up_name = user_info.get('name', f'UID_{uid}')
        print(f"\n{'='*60}")
        print(f"目标UP主: {up_name} (UID: {uid})")
        print(f"粉丝数: {user_info.get('follower', 'N/A')}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"获取UP主信息失败: {e}")
        up_name = f"UID_{uid}"
    all_videos = []
    pn = 1  # page number
    ps = 10  # page size, 最大通常是50
    print("[*] 开始获取视频列表...")
    while True:
        try:
            # 修复点：使用 pn 和 ps 参数代替旧的 page
            res = sync(u.get_videos(pn=pn, ps=ps))
            # 新版API返回结构可能不同，需要适配
            # 常见结构: res['list']['vlist'] 或直接 res 包含列表
            vlist = []
            if 'list' in res and 'vlist' in res['list']:
                vlist = res['list']['vlist']
            elif isinstance(res, list):
                vlist = res
            elif 'data' in res and 'list' in res['data']:
                vlist = res['data']['list']['vlist']
            if not vlist:
                print(f"[✓] 第 {pn} 页无数据，获取完成")
                break
            for v in vlist:
                all_videos.append({
                    'bvid': v.get('bvid'),
                    'title': v.get('title'),
                    'created': v.get('created'),      # 时间戳
                    'length': v.get('length'),          # 时长
                    'play': v.get('play', 0),           # 播放量（列表接口）
                    'comment': v.get('comment', 0),     # 评论数
                    'description': v.get('description', '')[:100],
                })
            print(f"  第 {pn} 页: 获取 {len(vlist)} 个视频，累计 {len(all_videos)} 个")
            # 检查是否最后一页 (根据返回的总数判断)
            total_count = 0
            if 'list' in res and 'page' in res['list']:
                total_count = res['list']['page'].get('count', 0)
            elif 'data' in res and 'page' in res['data']:
                total_count = res['data']['page'].get('count', 0)
            if total_count > 0 and len(all_videos) >= total_count:
                break
            # 如果返回数量小于请求数量，说明是最后一页
            if len(vlist) < ps:
                break
            pn += 1
            time.sleep(delay)
        except Exception as e:
            print(f"[✗] 获取第 {pn} 页失败: {e}")
            break
    print(f"[✓] 列表获取完成，共 {len(all_videos)} 个视频")
    return all_videos, up_name
def enrich_video_details(videos, delay=1.5):
    """
    获取每个视频的详细统计数据
    """
    enriched = []
    total = len(videos)
    print(f"\n[*] 开始获取视频详细数据（共 {total} 个）...")
    for idx, v in enumerate(videos, 1):
        bvid = v['bvid']
        try:
            v_obj = video.Video(bvid=bvid)
            info = sync(v_obj.get_info())
            stat = info.get('stat', {})
            enriched.append({
                'BV号': bvid,
                '标题': v['title'],
                '发布时间': datetime.fromtimestamp(v['created']).strftime('%Y-%m-%d %H:%M') if v['created'] else 'N/A',
                '时长': v['length'],
                '播放量': stat.get('view', 0),
                '点赞': stat.get('like', 0),
                '投币': stat.get('coin', 0),
                '收藏': stat.get('favorite', 0),
                '评论': stat.get('reply', 0),
                '分享': stat.get('share', 0),
                '弹幕': stat.get('danmaku', 0),
                '综合得分': stat.get('view', 0) + stat.get('like', 0)*10 + stat.get('reply', 0)*5,
                '链接': f"https://www.bilibili.com/video/{bvid}",
                '简介': v.get('description', '')
            })
            if idx % 10 == 0 or idx == total:
                print(f"  进度: {idx}/{total} ({idx/total*100:.1f}%)")
            time.sleep(delay)
        except Exception as e:
            print(f"[✗] 获取 {bvid} 详情失败: {e}")
            enriched.append({
                'BV号': bvid,
                '标题': v['title'],
                '发布时间': datetime.fromtimestamp(v['created']).strftime('%Y-%m-%d %H:%M') if v['created'] else 'N/A',
                '时长': v['length'],
                '播放量': v.get('play', 0),
                '点赞': 0, '投币': 0, '收藏': 0,
                '评论': v.get('comment', 0),
                '分享': 0, '弹幕': 0,
                '综合得分': 0,
                '链接': f"https://www.bilibili.com/video/{bvid}",
                '简介': v.get('description', ''),
                '备注': '数据获取失败'
            })
            time.sleep(delay)
    print(f"[✓] 详细数据获取完成")
    return enriched
def save_to_excel(data, up_name, min_likes=1000, min_comments=500):
    """保存到Excel"""
    df = pd.DataFrame(data)
    if df.empty:
        print("[!] 没有数据可保存")
        return None
    numeric_cols = ['播放量', '点赞', '投币', '收藏', '评论', '分享', '弹幕', '综合得分']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    high_engagement = df[
        (df['点赞'] >= min_likes) | (df['评论'] >= min_comments)
    ].copy()
    by_likes = df.sort_values('点赞', ascending=False).reset_index(drop=True)
    by_comments = df.sort_values('评论', ascending=False).reset_index(drop=True)
    by_play = df.sort_values('播放量', ascending=False).reset_index(drop=True)
    by_score = df.sort_values('综合得分', ascending=False).reset_index(drop=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = "".join(c for c in up_name if c.isalnum() or c in (' ', '_')).rstrip()
    filename = f"B站_{safe_name}_{timestamp}.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        by_likes.to_excel(writer, sheet_name='全部视频-按点赞', index=False)
        if not high_engagement.empty:
            high_engagement.sort_values('点赞', ascending=False).to_excel(
                writer, sheet_name='高互动视频', index=False
            )
        by_comments.to_excel(writer, sheet_name='按评论排序', index=False)
        by_play.to_excel(writer, sheet_name='按播放量排序', index=False)
        by_score.to_excel(writer, sheet_name='按综合得分排序', index=False)
        summary = pd.DataFrame([{
            'UP主': up_name,
            '视频总数': len(df),
            '高互动视频数': len(high_engagement),
            '总播放量': df['播放量'].sum(),
            '总点赞': df['点赞'].sum(),
            '总评论': df['评论'].sum(),
            '平均点赞': df['点赞'].mean(),
            '平均评论': df['评论'].mean(),
            '最高点赞视频': by_likes.iloc[0]['标题'] if len(by_likes) > 0 else 'N/A',
            '最高点赞数': by_likes.iloc[0]['点赞'] if len(by_likes) > 0 else 0,
        }])
        summary.to_excel(writer, sheet_name='数据汇总', index=False)
    print(f"\n{'='*60}")
    print(f"[✓] Excel 文件已保存: {filename}")
    print(f"  - 全部视频: {len(df)} 个")
    print(f"  - 高互动视频: {len(high_engagement)} 个")
    print(f"{'='*60}")
    return filename
def main():
    # ==================== 配置区域 ====================
    UP_UID = 703186600  # 麻薯波比呀的UID
    MIN_LIKES = 1000      # 最低点赞数
    MIN_COMMENTS = 500    # 最低评论数
    LIST_DELAY = 4.5      # 从1.0增加到2.5秒
    DETAIL_DELAY = 3.5    # 获取详情时的间隔
    SKIP_DETAILS = False   # 是否跳过详情获取
    # =================================================
    print(f"[*] 开始收集 UID: {UP_UID}")
    videos, up_name = get_up_all_videos(UP_UID, delay=LIST_DELAY)
    if not videos:
        print("[!] 未获取到任何视频，请检查UID是否正确")
        # 即使没有视频，也可以尝试检查UP主是否存在
        print("[*] 提示：如果UP主设置了隐藏投稿，可能无法获取列表")
        return
    if SKIP_DETAILS:
        print("\n[*] 跳过详情获取，使用列表基础数据...")
        simple_data = []
        for v in videos:
            simple_data.append({
                'BV号': v['bvid'],
                '标题': v['title'],
                '发布时间': datetime.fromtimestamp(v['created']).strftime('%Y-%m-%d %H:%M') if v['created'] else 'N/A',
                '时长': v['length'],
                '播放量': v.get('play', 0),
                '点赞': 0,
                '投币': 0,
                '收藏': 0,
                '评论': v.get('comment', 0),
                '分享': 0,
                '弹幕': 0,
                '综合得分': 0,
                '链接': f"https://www.bilibili.com/video/{v['bvid']}",
                '简介': v.get('description', ''),
                '备注': '仅列表数据'
            })
        final_data = simple_data
    else:
        final_data = enrich_video_details(videos, delay=DETAIL_DELAY)
    save_to_excel(final_data, up_name, min_likes=MIN_LIKES, min_comments=MIN_COMMENTS)
    print("\n[✓] 全部完成！")
if __name__ == "__main__":
    main()
```

# 二）开源工具
#### MediaCrawler - 自媒体平台爬虫 
https://github.com/NanmiCoder/MediaCrawler
安装：
```
# 进入项目目录
cd MediaCrawler
# 使用 uv sync 命令来保证 python 版本和相关依赖包的一致性
uv sync
# 仅在标准 Playwright 模式下需要安装浏览器驱动
uv run playwright install
```
启动： uv run uvicorn api.main:app --host 0.0.0.0 --port 1212 --reload
![[MediaCrawler.png]]
界面上需要输入关注的up的账号 和 cookie值
选择输出文件格式 csv 或者 json， 占时不支持excel
![[任务结束日志.png]]
配置项：
```bash
/home/qyc/gitee/MediaCrawler/config# cat bilibili_config.py
/home/qyc/gitee/MediaCrawler/config# cat  base_config.py
# 设置为 False 可以保持浏览器运行，方便调试
AUTO_CLOSE_BROWSER = True
SAVE_DATA_OPTION = "jsonl" 
SAVE_DATA_PATH = "data/"
```
数据保存在：
```bash
root@maizi:/home/qyc/gitee/MediaCrawler/data/bili/jsonl# ls
creator_comments_2026-04-21.jsonl  creator_contents_2026-04-21.jsonl  creator_creators_2026-04-21.jsonl

```


#### opencli 将界面爬虫 命令化
项目地址 https://github.com/jackwener/opencli
说明：
1. 使用内置适配器，适用于 Bilibili、知乎、小红书、Reddit、HackerNews、Twitter/X 等网站
2. 让 AI 代理操作任何网站——在你的 AI 代理（Claude Code、Cursor 等）中安装 `opencli-adapter-author` 技能，它就可以通过 `opencli browser` 原语在你的登录浏览器中导航、点击、输入/填充、提取和检查任何页面。

一旦安装了 `opencli-adapter-author` ，您的 AI 代理可以：
1. **Navigate** to any URL using your logged-in browser  
    导航到任何 URL，使用您的登录浏览器
2. **Read** page content via structured DOM snapshots (not screenshots)  
    通过结构化 DOM 快照（而非截图）读取页面内容
3. **Interact** — click buttons, fill forms, select options, press keys  
    交互 — 点击按钮，填写表单，选择选项，按键盘
4. **Extract** data from the page or intercept network API responses  
    从页面提取数据或拦截网络 API 响应
5. **Wait** for elements, text, or page transitions  
    等待元素、文本或页面转换
    
安装：
```
node --version
npm install -g @jackwener/opencli
# 安装浏览器插件 下载 opencli-extension-v1.0.6.zip，解压，在浏览器里安装插件
opencli doctor  # 检查安装状态
opencli bilibili hot --limit 5  # 获取B站最热门的前五视频


### Install skills 安装技能
npx skills add jackwener/opencli
```



### 三） 网页工具
**抖音热点宝:**  https://douhot.douyin.com

**飞瓜数据B站版**：
老牌的数据分析平台，功能很全面。提供热门视频、UP主涨粉等榜单，可以按分区、时间等维度筛选，支持实时监测视频数据变化和竞品对比
https://bz.feigua.cn/member#/ContentV2/WorkBench

**新榜有数**：覆盖B站、抖音、小红书等多平台，提供爆款内容榜和热门话题数据，适合进行跨平台的内容趋势分析
https://www.newrank.cn
