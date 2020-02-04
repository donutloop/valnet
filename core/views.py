from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from core.serializers import AddressSerializer
from core.usecase.predictor import Predictor


class ValidateView(APIView):
    serializer_class = AddressSerializer

    def __init__(self):
        super().__init__()
        self.predictor = Predictor()

    def post(self, request, format=None):
        address_serializer = AddressSerializer(data=request.data)
        address_serializer.is_valid(raise_exception=True)
        address = address_serializer.data

        return Response(self.predictor.validateAddress(address.get("address")))
