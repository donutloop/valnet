from django.db import models


class AddressValidationHistory(models.Model):
    address = models.CharField(max_length=2000)
    accuracy = models.FloatField()
    valid = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
