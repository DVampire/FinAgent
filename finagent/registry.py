from mmengine.registry import Registry

DATASET = Registry('data', locations=['finagent.data'])
PROMPT = Registry('prompt', locations=['finagent.prompt'])
AGENT = Registry('agent', locations=['finagent.agent'])
PROVIDER = Registry('provider', locations=['finagent.provider'])
DOWNLOADER = Registry('downloader', locations=['finagent.downloader'])
PROCESSOR = Registry('processor', locations=['finagent.processor'])
ENVIRONMENT = Registry('environment', locations=['finagent.environment'])
MEMORY = Registry('memory', locations=['finagent.memory'])
PLOTS = Registry('plot', locations=['finagent.plots'])