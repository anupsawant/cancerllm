from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .llmutils import LLM

class ChatView(APIView):
    def post(self, request, *args, **kwargs):
        # Get the client's IP address from the request's META dictionary
        client_ip = request.META.get('REMOTE_ADDR', None)

        # Assuming you want to receive JSON data in the request body
        data = request.data
        llm = LLM(str(client_ip))
        # Your custom logic to process the data from the request
        # For demonstration, we'll just echo back the received data along with the IP
        response_data = {
            'response': llm.run(data['query'])
        }

        return Response(response_data, status=status.HTTP_201_CREATED)