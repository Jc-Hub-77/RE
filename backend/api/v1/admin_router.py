# backend/api/v1/admin_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.schemas import admin_schemas, user_schemas, strategy_schemas # Added strategy_schemas
from backend.services import admin_service, user_service, strategy_service # Added strategy_service
from backend.dependencies import get_current_active_admin_user, get_current_active_user # Added get_current_active_user for consistency if needed
from backend.models import User, UserStrategySubscription # Added UserStrategySubscription for counts
from backend.db import get_db
from fastapi.responses import JSONResponse

router = APIRouter()

# --- Admin User Management Endpoints ---
@router.get("/users", response_model=admin_schemas.AdminUserListResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_list_users(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search_term: Optional[str] = Query(None),
    sort_by: str = Query("id", enum=["id", "username", "email", "created_at"]),
    sort_order: str = Query("asc", enum=["asc", "desc"])
):
    result = admin_service.list_all_users(db, page, per_page, search_term, sort_by, sort_order)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Error listing users"))
    return result

@router.post("/users/set-admin-status", response_model=user_schemas.GeneralResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_set_user_admin_status(
    request_data: admin_schemas.AdminSetAdminStatusRequest,
    db: Session = Depends(get_db)
):
    result = admin_service.set_user_admin_status(db, request_data.user_id, request_data.make_admin)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.post("/users/toggle-email-verified", response_model=user_schemas.GeneralResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_toggle_user_email_verified( # This function was missing in admin_service.py, assuming it will be added or user_service is used
    request_data: admin_schemas.AdminSetAdminStatusRequest, # Schema re-used, 'make_admin' interpreted as 'set_verified'
    db: Session = Depends(get_db)
):
    # Assuming a service function like: admin_service.toggle_user_email_verified_status(db, user_id, set_verified_status)
    # For now, let's assume it exists or will be created in admin_service.py
    # If it's in user_service, it needs to be callable by an admin.
    # Placeholder call, as service function was not in provided admin_service.py
    # result = user_service.admin_set_email_verified(db, request_data.user_id, request_data.make_admin)
    # This endpoint might need a dedicated service function in admin_service.py
    # For now, this will likely fail if admin_service.toggle_user_email_verified is not defined.
    # Let's assume admin_service.toggle_user_email_verified will be created.
    if not hasattr(admin_service, 'toggle_user_email_verified'):
         raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Email verification toggle service not implemented.")
    result = admin_service.toggle_user_email_verified(db, request_data.user_id, request_data.make_admin) # make_admin used as bool for verified
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result


@router.post("/users/toggle-active-status", response_model=user_schemas.GeneralResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_toggle_user_active_status(
    request_data: admin_schemas.AdminSetAdminStatusRequest,
    db: Session = Depends(get_db)
):
    # This correctly calls user_service.toggle_user_active_status which is fine if it has admin checks or is admin-specific.
    # Alternatively, admin_service could wrap this.
    result = user_service.toggle_user_active_status(db, request_data.user_id, request_data.make_admin)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result


# --- Admin Strategy Management Endpoints ---
@router.get("/managed-strategies", response_model=admin_schemas.AdminStrategyListResponse, dependencies=[Depends(get_current_active_admin_user)]) # Renamed path for clarity
async def admin_list_managed_strategies(db: Session = Depends(get_db)): # Renamed function for clarity
    result = admin_service.list_all_strategies_admin(db)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Error listing strategies"))
    return result

@router.post("/managed-strategies", response_model=admin_schemas.AdminActionResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_current_active_admin_user)])
async def admin_add_managed_strategy( # Renamed function for clarity
    strategy_data: admin_schemas.AdminStrategyCreateRequest,
    db: Session = Depends(get_db)
):
    result = admin_service.add_new_strategy_admin(
        db_session=db, name=strategy_data.name, description=strategy_data.description,
        python_code_path=strategy_data.python_code_path, default_parameters=strategy_data.default_parameters,
        category=strategy_data.category, risk_level=strategy_data.risk_level
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.put("/managed-strategies/{strategy_id}", response_model=admin_schemas.AdminActionResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_update_managed_strategy( # Renamed function for clarity
    strategy_id: int,
    strategy_update_data: admin_schemas.AdminStrategyUpdateRequest,
    db: Session = Depends(get_db)
):
    updates = strategy_update_data.dict(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided.")
    result = admin_service.update_strategy_admin(db, strategy_id, updates)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    elif result["status"] == "info":
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    return result

# --- Admin Subscription Management Endpoints ---
@router.get("/all-subscriptions", response_model=admin_schemas.AdminSubscriptionListResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_list_all_subscriptions(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user_id: Optional[int] = Query(None, description="Filter by User ID"),
    strategy_id: Optional[int] = Query(None, description="Filter by Strategy ID"),
    is_active: Optional[bool] = Query(None, description="Filter by active status")
):
    result = admin_service.list_all_subscriptions_admin(
        db,
        page=page,
        per_page=per_page,
        user_id=user_id,
        strategy_id=strategy_id,
        is_active=is_active
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result.get("message", "Error listing all subscriptions"))
    return result

@router.post("/subscriptions/{subscription_id}/deactivate", response_model=user_schemas.GeneralResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_deactivate_subscription(
    subscription_id: int,
    db: Session = Depends(get_db)
):
    # user_id is not strictly needed if by_admin=True bypasses ownership check in service
    result = strategy_service.deactivate_strategy_subscription(db_session=db, user_id=None, subscription_id=subscription_id, by_admin=True)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result

@router.put("/subscriptions/{subscription_id}/details", response_model=user_schemas.GeneralResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_update_subscription_details_route( # Renamed to avoid conflict
    subscription_id: int,
    update_data: admin_schemas.AdminSubscriptionUpdateRequest,
    db: Session = Depends(get_db)
):
    result = strategy_service.admin_update_subscription_details(
        db_session=db,
        subscription_id=subscription_id,
        new_status_message=update_data.new_status_message,
        new_is_active=update_data.new_is_active,
        new_expires_at_str=update_data.new_expires_at_str
    )
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"])
    return result


# --- Admin Site Settings Endpoint ---
@router.get("/site-settings", response_model=admin_schemas.AdminSiteSettingsResponse, dependencies=[Depends(get_current_active_admin_user)])
async def admin_get_site_settings():
    result = admin_service.get_site_settings_admin()
    return result

# --- Admin Dashboard Data ---
@router.get("/dashboard-summary", response_model=admin_schemas.AdminDashboardSummaryResponse, dependencies=[Depends(get_current_active_admin_user)]) # Define a schema for this
async def admin_dashboard_summary_route(db: Session = Depends(get_db)): # Renamed function
    total_users_data = admin_service.list_all_users(db, page=1, per_page=1)
    total_users = total_users_data.get("total_users", 0)

    total_revenue = admin_service.get_total_revenue(db)

    # Fetch active subscriptions count
    # A more direct count function in service might be better, but this works:
    active_subscriptions_data = admin_service.list_all_subscriptions_admin(db, page=1, per_page=1) # Assuming this returns total_subscriptions
    total_active_subscriptions = 0
    if active_subscriptions_data.get("subscriptions"): # Need to iterate and count active ones if not directly provided
        # This is inefficient. Add a dedicated count function in admin_service if this is slow.
        # For now, assuming total_subscriptions from list_all_subscriptions_admin is all subs, not just active.
        # Let's make a placeholder assumption or add a new service call.
        # For now, let's count from the first page of results if small number.
        # This should be properly implemented in admin_service.get_dashboard_summary_data()

        # Placeholder: A proper service function should provide this directly.
        # This is inefficient for many subscriptions.
        all_subs_for_count = db.query(UserStrategySubscription).filter(UserStrategySubscription.is_active == True).count()
        total_active_subscriptions = all_subs_for_count

    strategies_data = admin_service.list_all_strategies_admin(db)
    total_strategies = len(strategies_data.get("strategies", []))


    summary_data = {
        "totalUsers": total_users,
        "totalRevenue": total_revenue, # Renamed from totalRevenueLast30d for clarity unless it's specifically 30d
        "activeSubscriptions": total_active_subscriptions,
        "totalStrategies": total_strategies
    }
    return {"status": "success", "summary": summary_data}

# Removed duplicate public strategy endpoints from admin_router.
# They should reside in strategy_router.py for public access.
# If admin needs a different view, it should be via admin-specific path and schema.
# The /admin/managed-strategies serves the admin view of strategies.
