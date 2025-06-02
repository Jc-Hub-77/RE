import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from fastapi import status, HTTPException # Added HTTPException

# Assuming your FastAPI app instance is in backend.main
from backend.main import app 
from backend.schemas import user_schemas # For creating mock user
from backend.dependencies import get_current_active_admin_user # To override

# Placeholder for a mock admin user
mock_admin_user_instance = user_schemas.User(
    id=1, 
    username="testadmin", 
    email="admin@example.com", 
    is_active=True, 
    is_admin=True,
    email_verified=True,
    # Add other fields as required by your User schema, e.g. created_at, profile
    created_at="2023-01-01T12:00:00Z", # Example ISO format
    profile=None # Or a mock profile object
)

class TestAdminRouterSubscriptionRestart(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    # --- Test Successful Restart ---
    @patch('backend.services.admin_service.restart_strategy_subscription')
    def test_admin_restart_subscription_success(self, mock_admin_service_restart):
        
        # Mock the service call to return success
        service_response_message = "Strategy restarted successfully by service."
        mock_admin_service_restart.return_value = {
            "status": "success", 
            "message": service_response_message,
            "details": {"celery_task_id": "some_task_id"} # Example detail
        }

        subscription_id = 123
        admin_user_id = mock_admin_user_instance.id # from the mock admin

        # Override the dependency to simulate an authenticated admin user
        app.dependency_overrides[get_current_active_admin_user] = lambda: mock_admin_user_instance
        
        response = self.client.post(f"/api/v1/admin/subscriptions/{subscription_id}/restart")
        
        # Clean up dependency override
        app.dependency_overrides.clear()

        # Assertions
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        expected_router_response = {
            "status": "success",
            "message": f"Restart signal sent for subscription {subscription_id}. Result: {service_response_message}",
            "subscription_id": subscription_id
        }
        self.assertEqual(response.json(), expected_router_response)
        mock_admin_service_restart.assert_called_once_with(
            db=unittest.mock.ANY, # db session is injected by Depends(get_db)
            subscription_id=subscription_id,
            admin_user_id=admin_user_id
        )

    # --- Test Service Error (e.g., Subscription Not Found) ---
    @patch('backend.services.admin_service.restart_strategy_subscription')
    def test_admin_restart_subscription_service_error(self, mock_admin_service_restart):
        service_error_message = "Subscription not found by service."
        mock_admin_service_restart.return_value = {
            "status": "error", 
            "message": service_error_message
        }
        
        subscription_id = 404 # Example ID
        
        app.dependency_overrides[get_current_active_admin_user] = lambda: mock_admin_user_instance
        
        response = self.client.post(f"/api/v1/admin/subscriptions/{subscription_id}/restart")
        
        app.dependency_overrides.clear()
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.json(), {"detail": service_error_message})
        mock_admin_service_restart.assert_called_once_with(
            db=unittest.mock.ANY,
            subscription_id=subscription_id,
            admin_user_id=mock_admin_user_instance.id
        )

    # --- Test Unauthorized Access (No Admin Credentials) ---
    def test_admin_restart_subscription_unauthorized(self):
        # Ensure no admin override is active for this test
        app.dependency_overrides.clear() 

        response = self.client.post("/api/v1/admin/subscriptions/789/restart")
        
        # FastAPI typically returns 401 if Depends security fails due to missing token
        # or 403 if token is present but invalid/insufficient permissions (though this depends on get_current_active_admin_user impl)
        # Given get_current_active_admin_user likely raises HTTPException(status.HTTP_401_UNAUTHORIZED, ...)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED) 
        # The detail might vary based on how get_current_active_admin_user handles it
        # self.assertIn("Not authenticated", response.json().get("detail", ""))


    # --- Test 501 Not Implemented if service function is missing ---
    # This tests the router's own check: if not hasattr(admin_service, 'restart_strategy_subscription')
    @patch('backend.api.v1.admin_router.admin_service') # Patch admin_service specifically where it's used in admin_router
    def test_admin_restart_subscription_service_function_missing(self, mock_router_admin_service):
        
        # Make the admin_service object used by the router *not* have the function
        del mock_router_admin_service.restart_strategy_subscription
        # Alternative: mock_router_admin_service.hasattr.side_effect = lambda name: False if name == 'restart_strategy_subscription' else True

        subscription_id = 501 # Example ID
        
        app.dependency_overrides[get_current_active_admin_user] = lambda: mock_admin_user_instance
        
        response = self.client.post(f"/api/v1/admin/subscriptions/{subscription_id}/restart")
        
        app.dependency_overrides.clear()
        
        self.assertEqual(response.status_code, status.HTTP_501_NOT_IMPLEMENTED)
        self.assertEqual(response.json(), {"detail": "Strategy subscription restart service not implemented."})


if __name__ == '__main__':
    unittest.main()
