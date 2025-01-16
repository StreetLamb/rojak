# Food ordering example

Demostrate orchestrating multiple agents in a food ordering workflow.

## Setup

Run Temporal locally:
```shell
temporal server start-dev
```

Ensure you have `OPENAI_API_KEY` in the .env file:
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

Start the worker:
```shell
python run_worker.py
```

In another terminal, run the client: 
```shell
python send_message.py
```