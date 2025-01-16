# Weather Example

Demostrates how to create a rojak `Session` to interact with a Weather agent that uses tool calling and `context_variables`.

Make sure temporal is running:
```shell
temporal server start-dev
```

Ensure you have `OPENAI_API_KEY` in the .env file:
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

Run the worker:
```shell
python run_worker.py
```

On another terminal, run script to start session and send a message:
```
python run_workflow.py
```