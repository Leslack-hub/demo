import asyncio
import aiohttp


async def download_page(session, url):
    async with session.get(url) as response:
        return await response.text()


async def main():
    urls = ["https://www.baidu.com", "https://www.bilibili.com", "https://www.douyin.com"]
    async with aiohttp.ClientSession() as session:
        tasks = [download_page(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result[:100])  # 打印每个网页的前 100 个字符


if __name__ == '__main__':
    asyncio.run(main())
