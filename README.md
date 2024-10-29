pip install transformers datasets tokenizers fastapi gitpython torch


### Generate code 
curl -X POST "http://localhost:8000/generate_code/" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a Python function to calculate the factorial of a number"}'


GitHub Webhook Setup
Set up a GitHub webhook to trigger the /webhook/ endpoint whenever changes are made to the main branch.

Go to your GitHub repository, then Settings > Webhooks > Add webhook.
Enter the URL: http://your_server_ip:8000/webhook/
Set Content type to application/json.
Choose Just the push event.
Add the webhook.

### test from cli
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))