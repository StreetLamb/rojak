# Weather Example

Demostrates how to create a rojak `Session` to interact with a Weather agent that uses tool calling and `context_variables`.

Make sure temporal is running:
```shell
temporal server start-dev
```

Run the worker:
```shell
python run_worker.py
```

On another terminal, run script to start session and send a message:
```
python run_workflow.py
```