from django.test import TestCase, Client
import json
from rest_framework import status
from django.urls import reverse

# initialize the APIClient app
client = Client()


class ValidateAddressTest(TestCase):
    """ Test module for inserting a new puppy """

    def setUp(self):
        self.valid_payload = {
            'address': 'Slack Technologies Limited 4th Floor, One Park Place Hatch Street Upper Dublin 2, Irlanda',
        }
        self.invalid_payload = {
            'address': 'For Customers and Authorized Users who use Workspaces established for Customers in the US and '
                       'Canada',
        }

    def test_valid_address(self):
        response = client.post(
            reverse('validate'),
            data=json.dumps(self.valid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_invalid_address(self):
        response = client.post(
            reverse('validate'),
            data=json.dumps(self.valid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
