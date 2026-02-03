const express = require('express');
const { chromium } = require('playwright-extra')
const StealthPlugin = require('puppeteer-extra-plugin-stealth')
const { v4: uuidv4 } = require('uuid');
const yaml = require('js-yaml');
const fs = require('fs').promises;
const path = require('path');

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
const app = express();
const port = parseInt(process.env.LISTEN_PORT) || 3000;

app.use(express.json());

let browserPool = {};
const maxBrowsers = parseInt(process.env.MAX_BROWSERS) || 16;
let waitingQueue = [];

const initializeBrowserPool = (size) => {
  for (let i = 0; i < size; i++) {
    browserPool[String(i)] = {
      browserId: null,
      status: 'empty',
      browser: null,  // actually context
      browser0: null,  // browser
      pages: {},
      lastActivity: Date.now()
    };
  }
};

const v8 = require('v8');

const processNextInQueue = async () => {
  const availableBrowserslot = Object.keys(browserPool).find(
    id => browserPool[id].status === 'empty'
  );

  if (waitingQueue.length > 0 && availableBrowserslot) {
    const nextRequest = waitingQueue.shift();
    try {
      const browserEntry = browserPool[availableBrowserslot];
      let browserId = uuidv4()
      browserEntry.browserId = browserId
      browserEntry.status = 'not';
      nextRequest.res.send({ availableBrowserslot: availableBrowserslot });
    } catch (error) {
      nextRequest.res.status(500).send({ error: 'Failed to allocate browser.' });
    }
  } else if (waitingQueue.length > 0) {

  }
};


const releaseBrowser = async (browserslot) => {
  const browserEntry = browserPool[browserslot];
  if (browserEntry && browserEntry.browser) {
    await browserEntry.browser.close();
    await browserEntry.browser0.close();
    browserEntry.browserId = null;
    browserEntry.status = 'empty';
    browserEntry.browser = null;
    browserEntry.browser0 = null;
    browserEntry.pages = {};
    browserEntry.lastActivity = Date.now();

    processNextInQueue();
  }
};

setInterval(async () => {
  const now = Date.now();
  for (const [browserslot, browserEntry] of Object.entries(browserPool)) {
    if (browserEntry.status === 'not' && now - browserEntry.lastActivity > 600000) {
      await releaseBrowser(browserslot);
    }
  }
}, 60000);

function findPageByPageId(browserId, pageId) {
  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (browserEntry && browserEntry.pages[pageId]) {
    return browserEntry.pages[pageId];
  }
  return null;
}

function findPagePrefixesWithCurrentMark(browserId, currentPageId) {
  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  let pagePrefixes = [];

  if (browserEntry) {
    console.log(`current page id:${currentPageId}`, typeof currentPageId)
    for (const pageId in browserEntry.pages) {

      const page = browserEntry.pages[pageId];
      const pageTitle = page.pageTitle;
      console.log(`iter page id:${pageId}`, typeof pageId)
      const isCurrentPage = pageId === currentPageId;
      const pagePrefix = `Tab ${pageId}${isCurrentPage ? ' (current)' : ''}: ${pageTitle}`;

      pagePrefixes.push(pagePrefix);
    }
  }

  return pagePrefixes.length > 0 ? pagePrefixes.join('\n') : null;
}

const { Mutex } = require("async-mutex");
const mutex = new Mutex();
app.post('/getBrowser', async (req, res) => {
  const { storageState, geoLocation } = req.body;
  const tryAllocateBrowser = () => {
    const availableBrowserslot = Object.keys(browserPool).find(
      id => browserPool[id].status === 'empty'
    );
    let browserId = null;
    if (availableBrowserslot) {
      browserId = uuidv4()
      browserPool[availableBrowserslot].browserId = browserId
    }
    return {availableBrowserslot, browserId};
  };

  const waitForAvailableBrowser = () => {
    return new Promise(resolve => {
      waitingQueue.push(request => resolve(request));
    });
  };

  const PLAYWRIGHT_BACKEND = process.env.PLAYWRIGHT_BACKEND;
  const BROWSERLESS_TOKEN = process.env.BROWSERLESS_TOKEN;
  const PROXY_PORT = process.env.PROXY_PORT;
  const BROWSERLESS_TARGET_HOST = process.env.BROWSERLESS_TARGET_HOST;

  // Acquire the mutex lock
  const release = await mutex.acquire();

  try {
    let {availableBrowserslot, browserId} = tryAllocateBrowser();
    if (!availableBrowserslot) {
      await waitForAvailableBrowser().then((id) => {
        availableBrowserslot = id;
      });
    }
    console.log(storageState);
    let browserEntry = browserPool[availableBrowserslot];
    if (!browserEntry.browser) {
      chromium.use(StealthPlugin())

      if (
          !PLAYWRIGHT_BACKEND ||
          PLAYWRIGHT_BACKEND === 'local'
      ) {
        new_browser = await chromium.launch({
          headless: true,
          chromiumSandbox: true
        });
      } else if (PLAYWRIGHT_BACKEND === 'browserless+proxy') {
        const wsEndpoint = `ws://localhost:${PROXY_PORT}/chromium/playwright?token=${BROWSERLESS_TOKEN}&timeout=120000`;
        new_browser = await chromium.connect(wsEndpoint, { timeout: 120000 });
      } else if (PLAYWRIGHT_BACKEND === 'browserless') {
        const wsEndpoint = `wss://${BROWSERLESS_TARGET_HOST}/chromium/playwright?token=${BROWSERLESS_TOKEN}`;
        new_browser = await chromium.connect(wsEndpoint);
      }
      else {
        throw new Error("Unknown PLAYWRIGHT_BACKEND value");
      }

      browserEntry.browser = await new_browser.newContext({
        viewport: {width: 1024, height: parseInt(process.env.PLAYWRIGHT_VIEWPORT_H || '768')},
        locale: 'en-US',  // Set the locale to English (US)
        geolocation: { latitude: 40.4415, longitude: -80.0125 },  // Coordinates for Pittsburgh, PA, USA
        permissions: ['geolocation'],  // Grant geolocation permissions
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',  // Example user agent
        acceptDownloads: true
      });
      browserEntry.browser0 = new_browser;
    }
    browserEntry.status = 'not';
    browserEntry.lastActivity = Date.now();
    console.log(`browserId: ${browserId}`)
    res.send({browserId: browserId});
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to get browser.' });
  } finally {
    // Release the mutex lock
    release();
  }
});

app.post('/closeBrowser', async (req, res) => {
  const { browserId } = req.body;

  if (!browserId) {
    return res.status(400).send({ error: 'Missing required field: browserId.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (!browserEntry || !browserEntry.browser) {
    return res.status(404).send({ error: 'Browser not found.' });
  }

  try {
    await browserEntry.browser.close();
    await browserEntry.browser0.close();

    browserEntry.browserId = null;
    browserEntry.pages = {};
    browserEntry.browser = null;
    browserEntry.browser0 = null;
    browserEntry.status = 'empty';
    browserEntry.lastActivity = null;

    if (waitingQueue.length > 0) {
      const nextRequest = waitingQueue.shift();
      const nextAvailableBrowserId = Object.keys(browserPool).find(
        id => browserPool[id].status === 'empty'
      );
      if (nextRequest && nextAvailableBrowserId) {
        browserPool[nextAvailableBrowserId].status = 'not';
        nextRequest(nextAvailableBrowserId);
      }
    }

    res.send({ message: 'Browser closed successfully.' });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to close browser.' });
  }
});

app.post('/openPage', async (req, res) => {
  const { browserId, url } = req.body;

  if (!browserId || !url) {
    return res.status(400).send({ error: 'Missing browserId or url.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  // const browserEntry = browserPool[browserId];
  if (!browserEntry || !browserEntry.browser) {
    return res.status(404).send({ error: 'Browser not found.' });
  }
  console.log(await browserEntry.browser.storageState());
  const setCustomUserAgent = async (page) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'userAgent', {
        get: () => 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      });
    });
  };
  try {
    const page = await browserEntry.browser.newPage({'timeout': 60000});
    
    // for proxy support
    const PLAYWRIGHT_BACKEND = process.env.PLAYWRIGHT_BACKEND;

    if (PLAYWRIGHT_BACKEND === 'browserless+proxy') {
        page.setDefaultTimeout(120000);
    page.setDefaultNavigationTimeout(120000);
    } 
    

    await setCustomUserAgent(page);  
    // Inject script to force downloads
    await page.addInitScript(() => {
      document.addEventListener('click', (e) => {
        const target = e.target.closest('a');
        if (target && target.href) {
          const url = target.href;
          
          // Extract the pathname without query parameters or hash
          let pathname;
          try {
            const urlObj = new URL(url);
            pathname = urlObj.pathname.toLowerCase();
          } catch {
            pathname = url.toLowerCase();
          }
          
          // Check if pathname actually ENDS with these extensions
          const downloadableExtensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.txt', '.csv'];
          const hasDownloadableExtension = downloadableExtensions.some(ext => {
            // Check if pathname ends with the extension
            // or has extension followed by query/hash (like .pdf?version=2)
            const lastPart = pathname.split('/').pop() || '';
            return lastPart.endsWith(ext) || lastPart.includes(ext + '?') || lastPart.includes(ext + '#');
          });
          
          // Check for URL patterns that typically serve PDFs without .pdf extension
          const urlLower = url.toLowerCase();
          const pdfUrlPatterns = [
            '/pdf/',           // arxiv.org/pdf/xxxx
            'arxiv.org/pdf',   // specifically for arxiv
            '/download/',      // common download endpoints
            '/export?format=pdf',
            '/fulltext',
            'viewdoc/download',
            '/pdfviewer/',
            '/get_pdf/',
            '/downloadpdf/',
            '/getpdf',
            '/fetch.php',
            '/download.php',
            '/file.php'
          ];
          const isPdfUrl = pdfUrlPatterns.some(pattern => urlLower.includes(pattern));
          
          // Force download if it matches any condition
          const shouldDownload = hasDownloadableExtension || isPdfUrl;
          
          if (shouldDownload && !target.hasAttribute('download')) {
            // Set download attribute with a filename hint
            if (isPdfUrl && !hasDownloadableExtension) {
              const urlPath = target.href.split('?')[0];
              const filename = urlPath.split('/').pop() || 'document';
              target.setAttribute('download', filename.includes('.') ? filename : `${filename}.pdf`);
            } else {
              target.setAttribute('download', '');
            }
          }
        }
      }, true);
    });
    await page.goto(url, { timeout: 60000, waitUntil: "networkidle"});
    const pageIdint = Object.keys(browserEntry.pages).length;
    const pageTitle = await page.title();
    const pageId = String(pageIdint);
    // Initialize download promises array
    const downloadPromises = [];
    browserEntry.pages[pageId] = {
      'pageId': pageId,
      'pageTitle': pageTitle,
      'page': page,
      'downloadedFiles': [],
      'downloadSources': [],
      'downloadPromises': downloadPromises  // Store reference to promises
    };
    browserEntry.lastActivity = Date.now();
    // Create download directory once
    const downloadPath = path.resolve(`./DownloadedFiles/${browserId}`);
    await fs.mkdir(downloadPath, { recursive: true });
    // Listen for downloads - EXACTLY like browserless example
    page.on('download', (download) => {
      console.log('Download started:', download.url());
      const filename = download.suggestedFilename();
      const filePath = path.join(downloadPath, filename);

      // Push the saveAs promise directly - don't await here
      downloadPromises.push(
        download.saveAs(filePath)
          .then(() => {
            console.log(`Download saved: ${filePath}`);
            browserEntry.pages[pageId].downloadedFiles.push(filePath);
            return filePath;
          })
          .catch(error => {
            console.error('Download save error:', error);
            throw error;
        })
      );
    });
      
    const userAgent = await page.evaluate(() => navigator.userAgent);
    console.log('USER AGENT:', userAgent);
    res.send({ browserId, pageId });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to open new page.' });
  }
});

function parseAccessibilityTree(nodes) {
  const IGNORED_ACTREE_PROPERTIES = [
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "hiddenRoot",
    "hidden",
    "controls",
    "labelledby",
    "describedby",
    "url"
  ];
  const IGNORED_ACTREE_ROLES = [
    "gridcell",
  ];

  let nodeIdToIdx = {};
  nodes.forEach((node, idx) => {
    if (!(node.nodeId in nodeIdToIdx)) {
      nodeIdToIdx[node.nodeId] = idx;
    }
  });
  let treeIdxtoElement = {};
  function dfs(idx, depth, parent_name) {
    let treeStr = "";
    let node = nodes[idx];
    let indent = "\t".repeat(depth);
    let validNode = true;
    try {

      let role = node.role.value;
      let name = node.name.value;
      let nodeStr = `${role} '${name}'`;
      if (!name.trim() || IGNORED_ACTREE_ROLES.includes(role) || (parent_name.trim().includes(name.trim()) && ["StaticText", "heading", "image", "generic"].includes(role))){
        validNode = false;
      } else{
        let properties = [];
        (node.properties || []).forEach(property => {
          if (!IGNORED_ACTREE_PROPERTIES.includes(property.name)) {
            properties.push(`${property.name}: ${property.value.value}`);
          }
        });

        if (properties.length) {
          nodeStr += " " + properties.join(" ");
        }
      }

      if (validNode) {
        treeIdxtoElement[Object.keys(treeIdxtoElement).length + 1] = node;
        treeStr += `${indent}[${Object.keys(treeIdxtoElement).length}] ${nodeStr}`;
      }
    } catch (e) {
      validNode = false;
    }
    for (let childNodeId of node.childIds) {
      if (Object.keys(treeIdxtoElement).length >= 300) {
        break;
      }

      if (!(childNodeId in nodeIdToIdx)) {
        continue;
      }

      let childDepth = validNode ? depth + 1 : depth;
      let curr_name = validNode ? node.name.value : parent_name;
      let childStr = dfs(nodeIdToIdx[childNodeId], childDepth, curr_name);
      if (childStr.trim()) {
        if (treeStr.trim()) {
          treeStr += "\n";
        }
        treeStr += childStr;
      }
    }
    return treeStr;
  }

  let treeStr = dfs(0, 0, 'root');
  return {treeStr, treeIdxtoElement};
}

async function getBoundingClientRect(client, backendNodeId) {
  try {
      // Resolve the node to get the RemoteObject
      const remoteObject = await client.send("DOM.resolveNode", {backendNodeId: parseInt(backendNodeId)});
      const remoteObjectId = remoteObject.object.objectId;

      // Call a function on the resolved node to get its bounding client rect
      const response = await client.send("Runtime.callFunctionOn", {
          objectId: remoteObjectId,
          functionDeclaration: `
              function() {
                  if (this.nodeType === 3) { // Node.TEXT_NODE
                      var range = document.createRange();
                      range.selectNode(this);
                      var rect = range.getBoundingClientRect().toJSON();
                      range.detach();
                      return rect;
                  } else {
                      return this.getBoundingClientRect().toJSON();
                  }
              }
          `,
          returnByValue: true
      });
      return response;
  } catch (e) {
      return {result: {subtype: "error"}};
  }
}

async function fetchPageAccessibilityTree(accessibilityTree) {
  let seenIds = new Set();
  let filteredAccessibilityTree = [];
  let backendDOMids = [];
  for (let i = 0; i < accessibilityTree.length; i++) {
      if (filteredAccessibilityTree.length >= 20000) {
          break;
      }
      let node = accessibilityTree[i];
      if (!seenIds.has(node.nodeId) && 'backendDOMNodeId' in node) {
          filteredAccessibilityTree.push(node);
          seenIds.add(node.nodeId);
          backendDOMids.push(node.backendDOMNodeId);
      }
  }
  accessibilityTree = filteredAccessibilityTree;
  return [accessibilityTree, backendDOMids];
}

async function fetchAllBoundingClientRects(client, backendNodeIds) {
  const fetchRectPromises = backendNodeIds.map(async (backendNodeId) => {
      return getBoundingClientRect(client, backendNodeId);
  });

  try {
      const results = await Promise.all(fetchRectPromises);
      return results;
  } catch (error) {
      console.error("An error occurred:", error);
  }
}

function removeNodeInGraph(node, nodeidToCursor, accessibilityTree) {
  const nodeid = node.nodeId;
  const nodeCursor = nodeidToCursor[nodeid];
  const parentNodeid = node.parentId;
  const childrenNodeids = node.childIds;
  const parentCursor = nodeidToCursor[parentNodeid];
  // Update the children of the parent node
  if (accessibilityTree[parentCursor] !== undefined) {
    // Remove the nodeid from parent's childIds
    const index = accessibilityTree[parentCursor].childIds.indexOf(nodeid);
    //console.log('index:', index);
    accessibilityTree[parentCursor].childIds.splice(index, 1);
    // Insert childrenNodeids in the same location
    childrenNodeids.forEach((childNodeid, idx) => {
      if (childNodeid in nodeidToCursor) {
        accessibilityTree[parentCursor].childIds.splice(index + idx, 0, childNodeid);
      }
    });
    // Update children node's parent
    childrenNodeids.forEach(childNodeid => {
      if (childNodeid in nodeidToCursor) {
        const childCursor = nodeidToCursor[childNodeid];
        accessibilityTree[childCursor].parentId = parentNodeid;
      }
    });
  }
  accessibilityTree[nodeCursor].parentId = "[REMOVED]";
}

function processAccessibilityTree(accessibilityTree, minRatio) {
  const nodeidToCursor = {};
  accessibilityTree.forEach((node, index) => {
    nodeidToCursor[node.nodeId] = index;
  });
  let count = 0;
  accessibilityTree.forEach(node => {
    if (node.union_bound === undefined) {
      removeNodeInGraph(node, nodeidToCursor, accessibilityTree);
      return;
    }
    const x = node.union_bound.x;
    const y = node.union_bound.y;
    const width = node.union_bound.width;
    const height = node.union_bound.height;

    // Invisible node
    if (width === 0 || height === 0) {
      removeNodeInGraph(node, nodeidToCursor, accessibilityTree);
      return;
    }

    const inViewportRatio = getInViewportRatio(
      parseFloat(x),
      parseFloat(y),
      parseFloat(width),
      parseFloat(height),
    );
    // if (inViewportRatio < 0.5) {
    if (inViewportRatio < minRatio) {
      count += 1;
      removeNodeInGraph(node, nodeidToCursor, accessibilityTree);
    }
  });
  console.log('number of nodes marked:', count);
  accessibilityTree = accessibilityTree.filter(node => node.parentId !== "[REMOVED]");
  return accessibilityTree;
}

function getInViewportRatio(elemLeftBound, elemTopBound, width, height, config) {
  const elemRightBound = elemLeftBound + width;
  const elemLowerBound = elemTopBound + height;

  const winLeftBound = 0;
  const winRightBound = 1024;
  const winTopBound = 0;
  const winLowerBound = parseInt(process.env.PLAYWRIGHT_VIEWPORT_H || '768');

  const overlapWidth = Math.max(
      0,
      Math.min(elemRightBound, winRightBound) - Math.max(elemLeftBound, winLeftBound),
  );
  const overlapHeight = Math.max(
      0,
      Math.min(elemLowerBound, winLowerBound) - Math.max(elemTopBound, winTopBound),
  );

  const ratio = (overlapWidth * overlapHeight) / (width * height);
  return ratio;
}

app.post('/getAccessibilityTree', async (req, res) => {
  const { browserId, pageId, currentRound } = req.body;
  if (!browserId || !pageId) {
    return res.status(400).send({ error: 'Missing browserId or pageId.' });
  }
  const pageEntry = findPageByPageId(browserId, pageId);
  if (!pageEntry) {
    return res.status(404).send({ error: 'pageEntry not found.' });
  }
  const page = pageEntry.page;
  if (!page) {
    return res.status(404).send({ error: 'Page not found.' });
  }
  try {
    console.time('FullAXTTime');
    const client = await page.context().newCDPSession(page);
    const response = await client.send('Accessibility.getFullAXTree');
    const [axtree, backendDOMids] = await fetchPageAccessibilityTree(response.nodes);
    console.log('finished fetching page accessibility tree')
    const boundingClientRects = await fetchAllBoundingClientRects(client, backendDOMids);;
    console.log('finished fetching bounding client rects')
    console.log('boundingClientRects:', boundingClientRects.length, 'axtree:', axtree.length);
    for (let i = 0; i < boundingClientRects.length; i++) {
      if (axtree[i].role.value === 'RootWebArea') {
        axtree[i].union_bound = [0.0, 0.0, 10.0, 10.0];
      } else {
        axtree[i].union_bound = boundingClientRects[i].result.value;
      }
    }
    const clone_axtree = processAccessibilityTree(JSON.parse(JSON.stringify(axtree)), -1.0); // no space pruning
    const pruned_axtree = processAccessibilityTree(axtree, 0.5);
    const fullTreeRes = parseAccessibilityTree(clone_axtree);  // full tree
    const {treeStr, treeIdxtoElement} = parseAccessibilityTree(pruned_axtree);  // pruned tree
    console.timeEnd('FullAXTTime');
    
    const WEB_SERVER_DEBUG = process.env.WEB_SERVER_DEBUG;
    if (WEB_SERVER_DEBUG == 'True') {
      console.log(treeStr); 
    }

    pageEntry['treeIdxtoElement'] = treeIdxtoElement;
    const accessibilitySnapshot = await page.accessibility.snapshot();

    const prefix = findPagePrefixesWithCurrentMark(browserId, pageId) || '';
    let yamlWithPrefix = `${prefix}\n${treeStr}`;

    // if (pageEntry['downloadedFiles'].length > 0) {
    //   if (pageEntry['downloadSources'].length < pageEntry['downloadedFiles'].length) {
    //     const source_name = pruned_axtree[0].name.value;
    //     while (pageEntry['downloadSources'].length < pageEntry['downloadedFiles'].length) {
    //       pageEntry['downloadSources'].push(source_name);
    //     }
    //   }
    //   const downloadedFiles = pageEntry['downloadedFiles'];
    //   yamlWithPrefix += `\n\nYou have successfully downloaded the following files:\n`;
    //   downloadedFiles.forEach((file, idx) => {
    //     yamlWithPrefix += `File ${idx + 1} (from ${pageEntry['downloadSources'][idx]}): ${file}\n`;
    //   }
    //   );
    // }

    const screenshotBuffer = await page.screenshot();
    const fileName = `${browserId}@@${pageId}@@${currentRound}.png`;
    const screenshotPath = './screenshots';
    const filePath = path.join(screenshotPath, fileName);

    // Ensure the download directory exists
    try {
      await fs.access(screenshotPath);
    } catch (error) {
      if (error.code === 'ENOENT') {
        await fs.mkdir(screenshotPath, { recursive: true });
      } else {
        console.error(`Failed to access download directory: ${error}`);
        return;
      }
    }
    //
    await fs.writeFile(filePath, screenshotBuffer);
    const boxed_screenshotBuffer = await getboxedScreenshot(
      page,
      browserId,
      pageId,
      currentRound,
      treeIdxtoElement
    );

    const currentUrl = page.url();
    const html = await page.content();
    res.send({ yaml: yamlWithPrefix, fulltree: fullTreeRes.treeStr, url: currentUrl, html: html, snapshot: accessibilitySnapshot, nonboxed_screenshot: screenshotBuffer.toString("base64"), boxed_screenshot: boxed_screenshotBuffer.toString("base64"), downloaded_file_path: pageEntry['downloadedFiles']});
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to get accessibility tree.' });
  }
});

async function getboxedScreenshot(
  page,
  browserId,
  pageId,
  currentRound,
  treeIdxtoElement
) {
  // filter treeIdxtoElement to only include elements that are interactive
  // (e.g., buttons, links, form elements, etc.)
  const interactiveElements = {};
  Object.keys(treeIdxtoElement).forEach(function (index) {
    var elementData = treeIdxtoElement[index];
    var role = elementData.role.value;
    if (
      role === "button" ||
      role === "link" ||
      role === "tab" ||
      role.includes("box")
    ) {
      interactiveElements[index] = elementData;
    }
  });

  await page.evaluate((interactiveElements) => {
    Object.keys(interactiveElements).forEach(function (index) {
      var elementData = interactiveElements[index];
      var unionBound = elementData.union_bound; // Access the union_bound object

      // Create a new div element to represent the bounding box
      var newElement = document.createElement("div");
      var borderColor = "#000000"; // Use your color function to get the color
      newElement.style.outline = `2px dashed ${borderColor}`;
      newElement.style.position = "fixed";

      // Use union_bound's x, y, width, and height
      newElement.style.left = unionBound.x + "px";
      newElement.style.top = unionBound.y + "px";
      newElement.style.width = unionBound.width + "px";
      newElement.style.height = unionBound.height + "px";

      newElement.style.pointerEvents = "none";
      newElement.style.boxSizing = "border-box";
      newElement.style.zIndex = 2147483647;
      newElement.classList.add("bounding-box");

      // Create a floating label to show the index
      var label = document.createElement("span");
      label.textContent = index;
      label.style.position = "absolute";

      // Adjust label position with respect to union_bound
      label.style.top = Math.max(-19, -unionBound.y) + "px";
      label.style.left = Math.min(Math.floor(unionBound.width / 5), 2) + "px";
      label.style.background = borderColor;
      label.style.color = "white";
      label.style.padding = "2px 4px";
      label.style.fontSize = "12px";
      label.style.borderRadius = "2px";
      newElement.appendChild(label);

      // Append the element to the document body
      document.body.appendChild(newElement);
    });
  }, interactiveElements); // Pass treeIdxtoElement here as a second argument

  // Optionally wait a bit to ensure the boxes are drawn
  await page.waitForTimeout(1000);

  // Take the screenshot
  const screenshotBuffer = await page.screenshot();

  // Define the file name and path
  const fileName = `${browserId}@@${pageId}@@${currentRound}_with_box.png`;
  const filePath = path.join("./screenshots", fileName);

  // Write the screenshot to a file
  await fs.writeFile(filePath, screenshotBuffer);

  await page.evaluate(() => {
    document.querySelectorAll(".bounding-box").forEach((box) => box.remove());
  });
  return screenshotBuffer;
}

async function adjustAriaHiddenForSubmenu(menuitemElement) {
  try {
    const submenu = await menuitemElement.$('div.submenu');
    if (submenu) {
      await submenu.evaluate(node => {
        node.setAttribute('aria-hidden', 'false');
      });
    }
  } catch (e) {
    console.log('Failed to adjust aria-hidden for submenu:', e);
  }
}

async function clickElement(click_locator, adjust_aria_label, x1, x2, y1, y2) {
  const elements = adjust_aria_label ? await click_locator.elementHandles() : await click_locator.all();
  
  if (elements.length > 1) {
    for (const element of elements) {
      // 检查并处理链接
      await element.evaluate(el => {
        if (el.tagName.toLowerCase() === 'a') {
          // 移除 target 属性以避免新窗口打开
          if (el.hasAttribute('target')) {
            el.setAttribute('target', '_self');
          }
          
          // 检查是否是 PDF 或其他可下载文件
          const href = el.href || '';
          
          // Extract pathname for proper extension checking
          let pathname;
          try {
            const urlObj = new URL(href);
            pathname = urlObj.pathname.toLowerCase();
          } catch {
            pathname = href.toLowerCase();
          }
          
          // Check if pathname actually ends with these extensions
          const downloadableExtensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.txt', '.csv'];
          const hasDownloadableExtension = downloadableExtensions.some(ext => {
            const lastPart = pathname.split('/').pop() || '';
            return lastPart.endsWith(ext) || lastPart.includes(ext + '?') || lastPart.includes(ext + '#');
          });
          
          // Check for URL patterns that typically serve PDFs
          const hrefLower = href.toLowerCase();
          const pdfUrlPatterns = [
            '/pdf/',
            'arxiv.org/pdf',
            '/download/',
            '/export?format=pdf',
            '/fulltext',
            'viewdoc/download',
            '/pdfviewer/',
            '/get_pdf/',
            '/downloadpdf/',
            '/getpdf',
            '/fetch.php',
            '/download.php',
            '/file.php'
          ];
          const isPdfUrl = pdfUrlPatterns.some(pattern => hrefLower.includes(pattern));
          
          const shouldDownload = hasDownloadableExtension || isPdfUrl;
          
          if (shouldDownload && !el.hasAttribute('download')) {
            // Set download attribute with filename hint
            if (isPdfUrl && !hasDownloadableExtension) {
              const urlPath = href.split('?')[0];
              const filename = urlPath.split('/').pop() || 'document';
              el.setAttribute('download', filename.includes('.') ? filename : `${filename}.pdf`);
            } else {
              el.setAttribute('download', '');
            }
          }
        }
      });
    }
    
    // 找到最近的元素并点击
    const targetX = (x1 + x2) / 2;
    const targetY = (y1 + y2) / 2;
    let closestElement = null;
    let closestDistance = Infinity;

    for (const element of elements) {
      const boundingBox = await element.boundingBox();
      if (boundingBox) {
        const elementCenterX = boundingBox.x + boundingBox.width / 2;
        const elementCenterY = boundingBox.y + boundingBox.height / 2;
        const distance = Math.sqrt(
          Math.pow(elementCenterX - targetX, 2) + Math.pow(elementCenterY - targetY, 2)
        );
        if (distance < closestDistance) {
          closestDistance = distance;
          closestElement = element;
        }
      }
    }
    
    await closestElement.click({ timeout: 5000, force: true });
    if (adjust_aria_label) {
      await adjustAriaHiddenForSubmenu(closestElement);
    }
  } else if (elements.length === 1) {
    // 处理单个元素
    await elements[0].evaluate(el => {
      if (el.tagName.toLowerCase() === 'a') {
        if (el.hasAttribute('target')) {
          el.setAttribute('target', '_self');
        }
        
        const href = el.href || '';
        
        // Extract pathname for proper extension checking
        let pathname;
        try {
          const urlObj = new URL(href);
          pathname = urlObj.pathname.toLowerCase();
        } catch {
          pathname = href.toLowerCase();
        }
        
        // Check if pathname actually ends with these extensions
        const downloadableExtensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.txt', '.csv'];
        const hasDownloadableExtension = downloadableExtensions.some(ext => {
          const lastPart = pathname.split('/').pop() || '';
          return lastPart.endsWith(ext) || lastPart.includes(ext + '?') || lastPart.includes(ext + '#');
        });
        
        // Check for URL patterns that typically serve PDFs
        const hrefLower = href.toLowerCase();
        const pdfUrlPatterns = [
          '/pdf/',
          'arxiv.org/pdf',
          '/download/',
          '/export?format=pdf',
          '/fulltext',
          'viewdoc/download',
          '/pdfviewer/',
          '/get_pdf/',
          '/downloadpdf/',
          '/getpdf',
          '/fetch.php',
          '/download.php',
          '/file.php'
        ];
        const isPdfUrl = pdfUrlPatterns.some(pattern => hrefLower.includes(pattern));
        
        const shouldDownload = hasDownloadableExtension || isPdfUrl;
        
        if (shouldDownload && !el.hasAttribute('download')) {
          // Set download attribute with filename hint
          if (isPdfUrl && !hasDownloadableExtension) {
            const urlPath = href.split('?')[0];
            const filename = urlPath.split('/').pop() || 'document';
            el.setAttribute('download', filename.includes('.') ? filename : `${filename}.pdf`);
          } else {
            el.setAttribute('download', '');
          }
        }
      }
    });
    
    await elements[0].click({ timeout: 5000, force: true });
    if (adjust_aria_label) {
      await adjustAriaHiddenForSubmenu(elements[0]);
    }
  } else {
    return false;
  }
  return true;
}

app.post('/performAction', async (req, res) => {
  const { browserId, pageId, actionName, targetId, targetElementType, targetElementName, actionValue, needEnter } = req.body;

  if (['click', 'type'].includes(actionName) && (!browserId || !actionName || !targetElementType || !pageId)) {
      return res.status(400).send({ error: 'Missing required fields.' });
  } else if (!browserId || !actionName || !pageId) {
      return res.status(400).send({ error: 'Missing required fields.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (!browserEntry || !browserEntry.browser) {
      return res.status(404).send({ error: 'Browser not found.' });
  }

  const pageEntry = browserEntry.pages[pageId];
  if (!pageEntry || !pageEntry.page) {
      return res.status(404).send({ error: 'Page not found.' });
  }
  try {
      const page = pageEntry.page;
      const treeIdxtoElement = pageEntry.treeIdxtoElement;
      let adjust_aria_label = false;
      if (targetElementType === 'menuitem' || targetElementType === 'combobox') {
        adjust_aria_label = true;
      }
      switch (actionName) {

          case 'click':
              let element = treeIdxtoElement[targetId];
              let clicked = false;
              let click_locator;
              
              try {
                  click_locator = await page.getByRole(targetElementType, { 
                      name: targetElementName, 
                      exact: true, 
                      timeout: 5000
                  });
                  clicked = await clickElement(click_locator, adjust_aria_label, 
                      element.union_bound.x, 
                      element.union_bound.x + element.union_bound.width, 
                      element.union_bound.y, 
                      element.union_bound.y + element.union_bound.height
                  );
              } catch (e) {
                  console.log(e);
                  clicked = false;
              }
              
              if (!clicked) {
                  const click_locator = await page.getByRole(targetElementType, { 
                      name: targetElementName
                  });
                  clicked = await clickElement(click_locator, adjust_aria_label, 
                      element.union_bound.x, 
                      element.union_bound.x + element.union_bound.width, 
                      element.union_bound.y, 
                      element.union_bound.y + element.union_bound.height
                  );
                  
                  if (!clicked) {
                      const targetElementNameStartWords = targetElementName.split(' ').slice(0, 3).join(' ');
                      const click_locator = await page.getByText(targetElementNameStartWords);
                      clicked = await clickElement(click_locator, adjust_aria_label, 
                          element.union_bound.x, 
                          element.union_bound.x + element.union_bound.width, 
                          element.union_bound.y, 
                          element.union_bound.y + element.union_bound.height
                      );
                      
                      if (!clicked) {
                          return res.status(400).send({ error: 'No clickable element found.' });
                      }
                  }
              }
              
              // await page.waitForTimeout(5000);
              await page.waitForTimeout(3000);
    
              // Wait for any pending downloads to complete
              const downloadPromises = browserEntry.pages[pageId].downloadPromises;
              if (downloadPromises && downloadPromises.length > 0) {
                  console.log(`Waiting for ${downloadPromises.length} downloads to complete...`);
                  try {
                      const results = await Promise.all(downloadPromises);
                      console.log('Downloads completed:', results);
                      // Clear the array for next time
                      browserEntry.pages[pageId].downloadPromises.length = 0;
                  } catch (error) {
                      console.error('Error waiting for downloads:', error);
                  }
              }
              
              await page.waitForTimeout(2000);
              break;
          case 'type':
              let type_clicked = false;
              let locator;
              let node = treeIdxtoElement[targetId];
              try{
                locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000}).first()
                type_clicked = await clickElement(locator, adjust_aria_label, node.union_bound.x, node.union_bound.x + node.union_bound.width, node.union_bound.y, node.union_bound.y + node.union_bound.height);
              } catch (e) {
                console.log(e);
                type_clicked = false;
              }
              if (!type_clicked) {
                locator = await page.getByRole(targetElementType, { name: targetElementName}).first()
                type_clicked = await clickElement(locator, adjust_aria_label, node.union_bound.x, node.union_bound.x + node.union_bound.width, node.union_bound.y, node.union_bound.y + node.union_bound.height);
                if (!type_clicked) {
                  locator = await page.getByPlaceholder(targetElementName).first();
                  type_clicked = await clickElement(locator, adjust_aria_label, node.union_bound.x, node.union_bound.x + node.union_bound.width, node.union_bound.y, node.union_bound.y + node.union_bound.height);
                  if (!type_clicked) {
                    return res.status(400).send({ error: 'No clickable element found.' });
                  }
                }
              }

              await page.keyboard.press('Control+A');
              await page.keyboard.press('Backspace');
              if (needEnter) {
                const newactionValue = actionValue + '\n';
                await page.keyboard.type(newactionValue);
              } else {
                await page.keyboard.type(actionValue);
              }
              break;
          case 'select':
              let menu_locator = await page.getByRole(targetElementType, { name: targetElementName, exact:true, timeout: 5000});
              await menu_locator.selectOption({ label: actionValue })
              await menu_locator.click();
              break;
          case 'scroll':
              if (actionValue === 'down') {
                  await page.evaluate(() => window.scrollBy(0, window.innerHeight));
              } else if (actionValue === 'up') {
                  await page.evaluate(() => window.scrollBy(0, -window.innerHeight));
              } else {
                  return res.status(400).send({ error: 'Unsupported scroll direction.' });
              }
              break;
          case 'goback':
              await page.goBack();
              break;
          // case 'goto':
          //     // await page.goto("https://www.google.com");
          //     await page.goto(actionValue, { timeout: 60000, waitUntil: "networkidle"});
          //     break;
          case 'goto':
            try {
                const urlToGoto = actionValue;
                
                // 检查是否是可能的下载链接
                let pathname;
                try {
                    const urlObj = new URL(urlToGoto);
                    pathname = urlObj.pathname.toLowerCase();
                } catch {
                    pathname = urlToGoto.toLowerCase();
                }
                
                const downloadableExtensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.txt', '.csv'];
                const hasDownloadableExtension = downloadableExtensions.some(ext => {
                    const lastPart = pathname.split('/').pop() || '';
                    return lastPart.endsWith(ext) || lastPart.includes(ext + '?') || lastPart.includes(ext + '#');
                });
                
                const urlLower = urlToGoto.toLowerCase();
                const pdfUrlPatterns = [
                    '/pdf/',
                    'arxiv.org/pdf',
                    '/download/',
                    '/export?format=pdf',
                    '/fulltext',
                    'viewdoc/download',
                    '/pdfviewer/',
                    '/get_pdf/',
                    '/downloadpdf/',
                    '/getpdf',
                    '/fetch.php',
                    '/download.php',
                    '/file.php'
                ];
                const isPdfUrl = pdfUrlPatterns.some(pattern => urlLower.includes(pattern));
                
                const shouldDownload = hasDownloadableExtension || isPdfUrl;
                
                if (shouldDownload) {
                    console.log('Detected downloadable URL, using fetch to download:', urlToGoto);
                    
                    // 使用 page.evaluate 和 fetch 来下载
                    const downloadResult = await page.evaluate(async (url) => {
                        try {
                            const response = await fetch(url);
                            const blob = await response.blob();
                            
                            // 创建下载链接
                            const blobUrl = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = blobUrl;
                            
                            // 从 URL 或 response headers 获取文件名
                            const contentDisposition = response.headers.get('content-disposition');
                            let filename = 'download';
                            if (contentDisposition) {
                                const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                                if (filenameMatch && filenameMatch[1]) {
                                    filename = filenameMatch[1].replace(/['"]/g, '');
                                }
                            } else {
                                filename = url.split('/').pop().split('?')[0] || 'download';
                            }
                            
                            a.download = filename;
                            a.style.display = 'none';
                            document.body.appendChild(a);
                            a.click();
                            
                            // 清理
                            setTimeout(() => {
                                document.body.removeChild(a);
                                URL.revokeObjectURL(blobUrl);
                            }, 100);
                            
                            return { success: true, filename };
                        } catch (e) {
                            return { success: false, error: e.message };
                        }
                    }, urlToGoto);
                    
                    if (downloadResult.success) {
                        console.log('File downloaded via fetch:', downloadResult.filename);
                        await page.waitForTimeout(3000);
                    } else {
                        console.log('Fetch download failed:', downloadResult.error);
                        // 回退到正常导航
                        await page.goto(urlToGoto, { timeout: 60000, waitUntil: "domcontentloaded" });
                    }
                    
                } else {
                    await page.goto(urlToGoto, { timeout: 60000, waitUntil: "networkidle" });
                }
                
            } catch (error) {
                // Check if it's an abort error due to download
                console.log("FTQ: hello in goto error")
                if (error.message && error.message.includes('net::ERR_ABORTED')) {
                    console.log('Navigation aborted due to download, checking for downloads...');
                    
                    // Wait a moment for download to register and start
                    await page.waitForTimeout(2000);
                    
                    // Check if there are pending downloads
                    const downloadPromises = pageEntry.downloadPromises;
                    if (downloadPromises && downloadPromises.length > 0) {
                        console.log(`Download in progress, ${downloadPromises.length} file(s) downloading...`);
                        
                        try {
                            // Wait for all downloads to complete (with timeout of 30 seconds per file)
                            await Promise.race([
                                Promise.all(downloadPromises),
                                new Promise((_, reject) => 
                                    setTimeout(() => reject(new Error('Download timeout')), 30000 * downloadPromises.length)
                                )
                            ]);
                            
                            console.log('Download completed successfully');
                            console.log('Downloaded files:', pageEntry.downloadedFiles);
                            
                            // Don't throw error - treat as successful navigation
                            // The download listener in openPage already handled:
                            // 1. Saving to ./DownloadedFiles/${browserId}
                            // 2. Adding to browserEntry.pages[pageId].downloadedFiles
                            
                        } catch (downloadError) {
                            console.log('Download wait error:', downloadError);
                            // Still don't throw - downloads might complete later
                        }
                    } else {
                        // No downloads detected, but still an abort - might be a timing issue
                        console.log('Navigation aborted but no downloads detected yet, waiting...');
                        await page.waitForTimeout(3000);
                        
                        // Check again after waiting
                        if (pageEntry.downloadPromises && pageEntry.downloadPromises.length > 0) {
                            try {
                                await Promise.all(pageEntry.downloadPromises);
                                console.log('Late download completed');
                            } catch (err) {
                                console.log('Late download error:', err);
                            }
                        } else {
                            // Re-throw if it's not a download-related abort
                            throw error;
                        }
                    }
                } else {
                    // Re-throw other errors
                    throw error;
                }
            }
            break;
          case 'restart':
              await page.goto("https://www.bing.com", { timeout: 60000, waitUntil: "networkidle"});
              // await page.goto(actionValue);
              break;
          case 'wait':
              await sleep(3000);
              break;
          default:
              return res.status(400).send({ error: 'Unsupported action.' });
      }
      browserEntry.lastActivity = Date.now();
      await sleep(3000);
      const currentUrl = page.url();
      console.log(`current url: ${currentUrl}`);
      res.send({ message: 'Action performed successfully.' });
  } catch (error) {
      console.error(error);
      res.status(500).send({ error: 'Failed to perform action.' });
  }
});

app.post('/gotoUrl', async (req, res) => {
  const { browserId, pageId, targetUrl } = req.body;

  if (!targetUrl) {
      return res.status(400).send({ error: 'Missing required fields.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (!browserEntry || !browserEntry.browser) {
      return res.status(404).send({ error: 'Browser not found.' });
  }
  const pageEntry = browserEntry.pages[pageId];
  if (!pageEntry || !pageEntry.page) {
      return res.status(404).send({ error: 'Page not found.' });
  }

  try {
      const page = pageEntry.page;
      console.log(`target url: ${targetUrl}`);
      await page.goto(targetUrl, { timeout: 60000 });
      browserEntry.lastActivity = Date.now();
      await sleep(3000);
      const currentUrl = page.url();
      console.log(`current url: ${currentUrl}`);
      res.send({ message: 'Action performed successfully.' });
  } catch (error) {
      console.error(error);
      res.status(500).send({ error: 'Failed to perform action.' });
  }
});

app.post('/takeScreenshot', async (req, res) => {
  const { browserId, pageId } = req.body;

  if (!browserId || !pageId) {
    return res.status(400).send({ error: 'Missing required fields: browserId, pageId.' });
  }

  const slot = Object.keys(browserPool).find(slot => browserPool[slot].browserId === browserId);
  const browserEntry = browserPool[slot]
  if (!browserEntry || !browserEntry.browser) {
    return res.status(404).send({ error: 'Browser not found.' });
  }

  const pageEntry = browserEntry.pages[pageId];
  if (!pageEntry || !pageEntry.page) {
    return res.status(404).send({ error: 'Page not found.' });
  }

  try {
    const page = pageEntry.page;
    const screenshotBuffer = await page.screenshot({ fullPage: true });

    res.setHeader('Content-Type', 'image/png');
    res.send(screenshotBuffer);
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to take screenshot.' });
  }
});

app.post('/loadScreenshot', (req, res) => {
  const { browserId, pageId, currentRound } = req.body;
  const fileName = `${browserId}@@${pageId}@@${currentRound}.png`;
  const filePath = path.join('./screenshots', fileName);

  res.sendFile(filePath, (err) => {
    if (err) {
      console.error(err);
      if (err.code === 'ENOENT') {
        res.status(404).send({ error: 'Screenshot not found.' });
      } else {
        res.status(500).send({ error: 'Error sending screenshot file.' });
      }
    }
  });
});

app.post("/gethtmlcontent", async (req, res) => {
  const { browserId, pageId, currentRound } = req.body;
  // if (!browserId || !pageId) {
  //   return res.status(400).send({ error: 'Missing browserId or pageId.' });
  // }
  const pageEntry = findPageByPageId(browserId, pageId);
  const page = pageEntry.page;
  // if (!page) {
  //   return res.status(404).send({ error: 'Page not found.' });
  // }
  try {
    const html = await page.content();
    const currentUrl = page.url();
    res.send({ html: html, url: currentUrl });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: "Failed to get html content." });
  }
});

app.post('/getFile', async (req, res) => {
  try {
    const { filename } = req.body;
    if (!filename) {
      return res.status(400).send({ error: 'Filename is required.' });
    }
    const data = await fs.readFile(filename);  // simply directly read it!
    const base64String = data.toString('base64');
    res.send({ file: base64String });
  } catch (err) {
    console.error(err);
    res.status(500).send({ error: 'File not found or cannot be read.' });
  }
});

app.listen(port, () => {
  initializeBrowserPool(maxBrowsers);
  console.log(`Server listening at http://localhost:${port}`);
});


process.on('exit', async () => {
  for (const browserEntry of browserPool) {
      await browserEntry.browser.close();
      await browserEntry.browser0.close();
  }
});