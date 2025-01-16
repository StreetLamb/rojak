# MCP Weather Example

Demostrate how a weather agent can call tools from a MCP server.

Make sure temporal is running:
```shell
temporal server start-dev
```

Ensure you have `OPENAI_API_KEY` in the .env file:
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

Run the script:
```shell
python main.py
```