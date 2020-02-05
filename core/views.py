from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from core.serializers import AddressSerializer
from core.usecase.predictor import Predictor
import logging
from django.http import HttpResponseServerError

logger = logging.getLogger(__name__)
predictor = Predictor()


class ValidateView(APIView):
    serializer_class = AddressSerializer

    def __init__(self):
        super().__init__()

    def post(self, request, format=None):
        address_serializer = AddressSerializer(data=request.data)
        address_serializer.is_valid(raise_exception=True)
        address = address_serializer.data

        valid = False
        try:
            valid = predictor.validateAddress(address.get("address"))
        except Exception:
            logger.error("could not validate address, value:{}".format(address), exc_info=True)
            return HttpResponseServerError()

        data = {
            "valid": valid,
        }

        return Response(data)
