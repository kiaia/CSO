# aiohttp_proxy_server.py
import os
from aiohttp import web, ClientSession, WSMsgType, ClientWSTimeout
import aiohttp
import asyncio
import logging
from urllib.parse import parse_qs, urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_proxy = os.environ.get("TAIJI_PROXY", "").strip()
PROXY = _proxy or None
TARGET_HOST = os.environ.get("BROWSERLESS_TARGET_HOST", "production-sfo.browserless.io")
BROWSERLESS_TOKEN = os.environ.get("BROWSERLESS_TOKEN", "")
FORWARD_PORT = int(os.environ.get("PROXY_PORT", "8765"))

async def websocket_handler(request):
    """Process WebSocket requests"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # path of the request
    path = request.path
    
    # clear parameters
    query_params = dict(request.query)
    
    # remove arguments like --keep-alive
    cleaned_params = {}
    for key, value in query_params.items():
        # filter out params starting with --
        if not key.startswith('--'):
            cleaned_params[key] = value
    
    # ensure token is passed
    if 'token' not in cleaned_params:
        cleaned_params['token'] = BROWSERLESS_TOKEN
    
    # clear the string
    query_string = urlencode(cleaned_params)
    
    # construct target url
    if '/chromium/playwright' in path:
        target_url = f"wss://{TARGET_HOST}{path}?{query_string}"
    else:
        # use default playwright 
        target_url = f"wss://{TARGET_HOST}/chromium/playwright?{query_string}"
    
    logger.info(f"Forwarding to: {target_url}")
    
    # set timeout
    timeout = ClientWSTimeout(
        ws_receive=650,  # a tad bit longer than 10 mins
        ws_close=20      # timeout for closing
    )
    
    session = ClientSession()
    try:
        # connect to remote WebSocket
        # p.s.: for wss:// connections，ssl should be True or default
        ws_connect_kwargs = dict(
            ssl=True,
            timeout=timeout,
            max_msg_size=0,
            headers={'User-Agent': 'Playwright/1.54.1'},  # updated User-Agent
        )
        if PROXY:
            ws_connect_kwargs["proxy"] = PROXY
        remote_ws = await session.ws_connect(target_url, **ws_connect_kwargs)
        
        logger.info("Connected to remote WebSocket")
        
        # forward both ways
        async def forward_to_remote():
            try:
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        await remote_ws.send_str(msg.data)
                        logger.debug(f"→ Remote: {msg.data[:100] if len(msg.data) <= 100 else msg.data[:100] + '...'}")
                    elif msg.type == WSMsgType.BINARY:
                        await remote_ws.send_bytes(msg.data)
                        logger.debug(f"→ Remote: {len(msg.data)} bytes")
                    elif msg.type == WSMsgType.ERROR:
                        logger.error(f'WebSocket error: {ws.exception()}')
                        break
                    elif msg.type == WSMsgType.CLOSE:
                        logger.info("Local WebSocket closed")
                        break
            except Exception as e:
                logger.error(f"Error forwarding to remote: {e}")
        
        async def forward_to_local():
            try:
                async for msg in remote_ws:
                    if msg.type == WSMsgType.TEXT:
                        await ws.send_str(msg.data)
                        logger.debug(f"← Local: {msg.data[:100] if len(msg.data) <= 100 else msg.data[:100] + '...'}")
                    elif msg.type == WSMsgType.BINARY:
                        await ws.send_bytes(msg.data)
                        logger.debug(f"← Local: {len(msg.data)} bytes")
                    elif msg.type == WSMsgType.ERROR:
                        logger.error(f'Remote WebSocket error: {remote_ws.exception()}')
                        break
                    elif msg.type == WSMsgType.CLOSE:
                        logger.info("Remote WebSocket closed")
                        break
            except Exception as e:
                logger.error(f"Error forwarding to local: {e}")
        
        # run the forwarding
        await asyncio.gather(
            forward_to_remote(),
            forward_to_local(),
            return_exceptions=True
        )
        
    except aiohttp.ClientError as e:
        logger.error(f"Client error: {e}")
        error_msg = f"Error connecting to remote WebSocket: {str(e)}"
        try:
            await ws.send_str(error_msg)
        except:
            pass
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'remote_ws' in locals():
            await remote_ws.close()
        await session.close()
        await ws.close()
    
    return ws

# create application
app = web.Application()
# capture all paths
app.router.add_get('/{path:.*}', websocket_handler)

if __name__ == '__main__':
    logger.info(f"Starting proxy server on http://localhost:{FORWARD_PORT}")
    logger.info(f"Proxying to {TARGET_HOST} via {PROXY}")
    web.run_app(app, host='0.0.0.0', port=FORWARD_PORT, access_log=logger)
