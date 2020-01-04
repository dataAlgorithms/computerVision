root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# python37 asyncio_aiohttp.py 
Results: [<ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7F39)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
, <ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7F83)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
, <ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7FA3)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
, <ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7EC9)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
, <ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7F82)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
, <ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947+gzip"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7F7F)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
, <ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7F16)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
, <ClientResponse(http://example.com) [200 OK]>
<CIMultiDictProxy('Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Cache-Control': 'max-age=604800', 'Content-Type': 'text/html; charset=UTF-8', 'Date': 'Sat, 04 Jan 2020 08:00:41 GMT', 'Etag': '"3147526947"', 'Expires': 'Sat, 11 Jan 2020 08:00:41 GMT', 'Last-Modified': 'Thu, 17 Oct 2019 07:18:26 GMT', 'Server': 'ECS (dcb/7EEC)', 'Vary': 'Accept-Encoding', 'X-Cache': 'HIT', 'Content-Length': '648')>
]
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# cat asyncio_aiohttp.py 
import aiohttp
import asyncio

async def get(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return response

loop = asyncio.get_event_loop()
coroutines = [get("http://example.com") for _ in range(8)]
results = loop.run_until_complete(asyncio.gather(*coroutines))
print("Results: %s" % results)
root@ubuntu:/home/zhouping/anaconda3/pyLearn/scalary# 
