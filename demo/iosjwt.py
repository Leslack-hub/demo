from appstoreserverlibrary.api_client import AppStoreServerAPIClient, APIException
from appstoreserverlibrary.models.Environment import Environment

key_id = "29WRJB2336"
issuer_id = "293FKJV8F8"
bundle_id = "com.nineton.shouzhang"
environment = Environment.SANDBOX

apikey = """
-----BEGIN PRIVATE KEY-----
MIGTAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBHkwdwIBAQQgBCkKz3d7yM4jfamR
3GmeNntlc9/UbmSsrugefA0qIYigCgYIKoZIzj0DAQehRANCAAQWOo9KafNFaHB5
D58S/UKxHoYPMjEyF4uCAtmpcV0gZ7ap1Ml93s37BBQLpmpdx5JYskGVlwVfs6Yd
dycO+3YL
-----END PRIVATE KEY-----
"""

client = AppStoreServerAPIClient(apikey.encode(), key_id, issuer_id, bundle_id, environment)

try:
    response = client.request_test_notification()
    print(response)
except APIException as e:
    print(e)
