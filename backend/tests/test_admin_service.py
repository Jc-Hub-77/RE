import unittest
from unittest.mock import patch, MagicMock, call  # Added call
import logging

# Assuming admin_service is in backend.services.admin_service
from backend.services import admin_service 
from backend.schemas import user_schemas # For potential User model if needed by mocks

# Configure logger for tests if necessary, or mock it
# logging.basicConfig(level=logging.INFO) # Example, usually not needed if mocking logger

class TestAdminServiceSubscriptionRestart(unittest.TestCase):

    @patch('backend.services.admin_service.live_trading_service.restart_strategy_admin')
    @patch('backend.services.admin_service.logger')
    def test_restart_strategy_subscription_success(self, mock_logger, mock_restart_admin):
        """
        Test successful delegation to live_trading_service.restart_strategy_admin.
        """
        mock_db_session = MagicMock()
        subscription_id = 123
        admin_user_id = 1

        expected_response = {"status": "success", "message": "Strategy restarted by live_trading_service"}
        mock_restart_admin.return_value = expected_response

        result = admin_service.restart_strategy_subscription(
            db_session=mock_db_session,
            subscription_id=subscription_id,
            admin_user_id=admin_user_id
        )

        # Assertions
        mock_logger.info.assert_any_call(
            f"Admin User ID: {admin_user_id} initiated restart for subscription ID: {subscription_id} via admin_service, delegating to live_trading_service."
        )
        mock_restart_admin.assert_called_once_with(
            db_session=mock_db_session,
            subscription_id=subscription_id,
            admin_id=admin_user_id 
        )
        self.assertEqual(result, expected_response)
        mock_logger.info.assert_any_call( # Check for the success log AFTER the call
            f"Admin service call to restart subscription ID {subscription_id} completed successfully. Result: {expected_response.get('message')}"
        )


    @patch('backend.services.admin_service.live_trading_service.restart_strategy_admin')
    @patch('backend.services.admin_service.logger')
    def test_restart_strategy_subscription_error_from_live_service(self, mock_logger, mock_restart_admin):
        """
        Test error propagation from live_trading_service.restart_strategy_admin.
        """
        mock_db_session = MagicMock()
        subscription_id = 456
        admin_user_id = 2

        error_response = {"status": "error", "message": "Live trading service failed to restart"}
        mock_restart_admin.return_value = error_response

        result = admin_service.restart_strategy_subscription(
            db_session=mock_db_session,
            subscription_id=subscription_id,
            admin_user_id=admin_user_id
        )

        # Assertions
        mock_logger.info.assert_any_call(
            f"Admin User ID: {admin_user_id} initiated restart for subscription ID: {subscription_id} via admin_service, delegating to live_trading_service."
        )
        mock_restart_admin.assert_called_once_with(
            db_session=mock_db_session,
            subscription_id=subscription_id,
            admin_id=admin_user_id
        )
        self.assertEqual(result, error_response)
        mock_logger.error.assert_called_once_with( # Check for the error log AFTER the call
            f"Admin service call to restart subscription ID {subscription_id} failed. Reason: {error_response.get('message')}"
        )

if __name__ == '__main__':
    unittest.main()
