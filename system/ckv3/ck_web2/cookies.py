from requests.cookies import RequestsCookieJar


# Cookie Configuration
# ====================
# This file contains cookie configurations for web browsing agents.
# 
# IMPORTANT: Do NOT commit real session cookies to the repository!
# 
# To use cookies:
# 1. Export your browser cookies using a browser extension
# 2. Convert them to the format below
# 3. Store them in a separate file (e.g., cookies_private.py)
# 4. Import from that file in your local environment
# 
# Example cookie format:
# {
#     "domain": ".example.com",
#     "expirationDate": 1234567890,
#     "hostOnly": False,
#     "httpOnly": False,
#     "name": "cookie_name",
#     "path": "/",
#     "sameSite": "lax",
#     "secure": True,
#     "session": False,
#     "storeId": None,
#     "value": "your_cookie_value_here",
# }

# Default empty cookie list - replace with your own cookies as needed
COOKIES_LIST = [
    # Add your cookies here
    # Example:
    # {
    #     "domain": ".youtube.com",
    #     "name": "CONSENT",
    #     "value": "your_value_here",
    #     "path": "/",
    #     "secure": True,
    # },
]


def load_cookies_from_file(filepath):
    """
    Load cookies from an external file for local use.
    
    Args:
        filepath: Path to a Python file containing COOKIES_LIST
        
    Returns:
        list: List of cookie dictionaries
        
    Example:
        cookies = load_cookies_from_file('cookies_private.py')
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("cookies_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COOKIES_LIST if hasattr(module, 'COOKIES_LIST') else []


def get_cookies_jar():
    """
    Convert COOKIES_LIST to a RequestsCookieJar object.
    
    Returns:
        RequestsCookieJar: Cookie jar for use with requests library
    """
    jar = RequestsCookieJar()
    for cookie in COOKIES_LIST:
        jar.set(
            name=cookie.get('name'),
            value=cookie.get('value'),
            domain=cookie.get('domain'),
            path=cookie.get('path', '/'),
        )
    return jar
