# User Authentication System Guide

## Overview

The Mangrove Carbon Detection system now includes a complete user authentication system with login, signup, and logout functionality. Users must authenticate before they can access the satellite image upload and carbon detection features.

## Features

### ✅ User Registration (Signup)
- Create new user accounts with username and password
- Passwords are hashed using Werkzeug security (`generate_password_hash`)
- Validation rules:
  - Username: minimum 3 characters, must be unique
  - Password: minimum 6 characters
  - Confirm password must match

### ✅ User Login
- Authenticate with username and password
- Secure password verification using `check_password_hash`
- Session management via Flask-Login

### ✅ Session Management
- Automatic session creation after signup
- Session persistence across page reloads
- Logout functionality to clear sessions

### ✅ Route Protection
- `/upload` endpoint requires authentication (`@login_required`)
- Unauthenticated users see login prompt instead of upload interface

## Backend API Endpoints

### POST `/signup`
Register a new user account

**Request:**
```json
{
  "username": "john_doe",
  "password": "securepass123",
  "confirm_password": "securepass123"
}
```

**Response (Success):**
```json
{
  "success": true,
  "username": "john_doe"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Username already exists"
}
```

**Validation Errors:**
- Missing fields: `All fields are required`
- Short username: `Username must be at least 3 characters`
- Short password: `Password must be at least 6 characters`
- Passwords don't match: `Passwords do not match`
- Username exists: `Username already exists` (HTTP 409)

---

### POST `/login`
Authenticate an existing user

**Request:**
```json
{
  "username": "john_doe",
  "password": "securepass123"
}
```

**Response (Success):**
```json
{
  "success": true,
  "username": "john_doe"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "User not found"
}
```

**Validation Errors:**
- Missing fields: `Username and password are required`
- Invalid user: `User not found` (HTTP 401)
- Wrong password: `Invalid password` (HTTP 401)

---

### POST `/logout`
Clear user session and logout

**Response:**
```json
{
  "success": true
}
```

---

### GET `/auth_status`
Check current authentication status

**Response (Authenticated):**
```json
{
  "authenticated": true,
  "username": "john_doe"
}
```

**Response (Not Authenticated):**
```json
{
  "authenticated": false,
  "username": null
}
```

## Frontend Components

### Login Modal
Located in `templates/index.html` (lines ~298-363)

**Features:**
- Toggle between login and signup modes
- Error message display
- Clean, responsive design with Tailwind CSS
- Auto-focus on first input field
- Form validation before submission

**HTML Elements:**
- `#loginModal` - Modal container
- `#loginForm` - Login form (hidden by default)
- `#signupForm` - Signup form (visible by default)
- `#authError` - Error message display

### User Menu
Located in navbar

**Shows when authenticated:**
- User initial avatar (first letter of username)
- Username display
- Logout button in dropdown menu

**Shows when not authenticated:**
- Login button that opens the modal

## JavaScript Functions

### `checkAuthStatus()`
Fetches current authentication status and updates UI accordingly.

```javascript
const authenticated = await checkAuthStatus();
```

### `updateUIForAuth(isAuthenticated, username)`
Updates UI elements based on authentication state.

```javascript
updateUIForAuth(true, "john_doe");
```

### `showAuthModal(mode)`
Opens the authentication modal in login or signup mode.

```javascript
showAuthModal('signup');  // Show signup form
showAuthModal('login');   // Show login form
```

### `hideAuthModal()`
Closes the authentication modal.

```javascript
hideAuthModal();
```

### `handleLogin()`
Handles login form submission.

```javascript
// Called automatically on form submit
// Gets username and password from form inputs
// Calls /login endpoint
// Updates UI on success
```

### `handleSignup()`
Handles signup form submission.

```javascript
// Called automatically on form submit
// Gets username, password, and confirm_password from form inputs
// Calls /signup endpoint
// Updates UI on success
```

### `handleLogout()`
Handles logout functionality.

```javascript
// Called on logout button click
// Calls /logout endpoint
// Updates UI
```

## User Flow

### First Time User (Signup)

1. **User arrives at app** → Not authenticated
2. **User clicks "Login" button** → Modal opens with signup form
3. **User fills signup form** → Username, password, confirm password
4. **User submits form** → API validates and creates account
5. **User auto-logged in** → Upload interface becomes visible
6. **User can now upload images** → For carbon detection

### Returning User (Login)

1. **User arrives at app** → Not authenticated
2. **User clicks "Login" button** → Modal opens with signup form by default
3. **User clicks "Login" link** → Switches to login form
4. **User enters credentials** → Username and password
5. **User submits form** → API validates credentials
6. **User logged in** → Upload interface becomes visible

### Logout

1. **User clicks user avatar** → Dropdown menu opens
2. **User clicks "Logout"** → Session cleared
3. **UI updates** → Login button appears, upload interface hidden

## Data Storage

### In-Memory Storage
Accounts are stored in-memory in a Python dictionary (`users_db`):

```python
users_db = {
    "username": "password_hash",
    "john_doe": "pbkdf2:sha256:600000$...",
    ...
}
```

**Warning:** This is ephemeral - all user accounts are lost when the server restarts.

### Production Recommendation
For production deployment, replace in-memory storage with:
- **SQLite** - Simple file-based database
- **PostgreSQL** - Production-grade database
- **MongoDB** - NoSQL option

## Security Notes

✅ **Implemented:**
- Passwords hashed with PBKDF2-SHA256 (Werkzeug's `generate_password_hash`)
- Session management via Flask-Login
- CSRF protection ready (Flask-Login compatible)
- Route protection with `@login_required`
- Input validation on both frontend and backend

⚠️ **Recommendations for Production:**
- Use HTTPS/TLS for all communications
- Implement rate limiting on `/login` and `/signup` endpoints
- Add email verification for signup
- Implement password reset functionality
- Use database instead of in-memory storage
- Add logging for authentication events
- Implement account lockout after failed attempts
- Use stronger SECRET_KEY (currently hardcoded)

## Testing the Authentication

### Test Signup
```bash
curl -X POST http://localhost:5000/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"test123","confirm_password":"test123"}'
```

### Test Login
```bash
curl -X POST http://localhost:5000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"test123"}'
```

### Test Auth Status
```bash
curl http://localhost:5000/auth_status
```

### Test Logout
```bash
curl -X POST http://localhost:5000/logout
```

## Configuration

### Flask Configuration
Located in `app.py` lines 42-47:

```python
app.config["SECRET_KEY"] = "mangrove-carbon-detection-secret-2024"
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
users_db = {}
```

### LoginManager Settings
- `login_view = "login"` - Redirects to login route if not authenticated
- `user_loader` - Callback to load User object from username

## Troubleshooting

### Issue: "User not found" error on login
**Solution:** Make sure the username was registered. Create a new account or check spelling.

### Issue: "Passwords do not match" on signup
**Solution:** Confirm password must exactly match the password field.

### Issue: User stays logged in after logout
**Solution:** Check browser cookies - clear cache and try again. Sessions are stored server-side.

### Issue: "Username already exists"
**Solution:** Choose a different username. Each username must be unique.

## Files Modified

### Backend
- `app.py` - Added Flask-Login setup, User class, authentication routes
- `requirements.txt` - Added `flask-login>=0.6.2` and `werkzeug>=2.3.0`

### Frontend
- `templates/index.html` - Updated login modal with signup/login forms
- `static/script.js` - Added authentication handlers and UI updates

## Next Steps

For production deployment:
1. Replace in-memory user storage with a database
2. Implement email verification
3. Add password reset functionality
4. Enable HTTPS/TLS
5. Configure rate limiting
6. Set up proper logging
7. Use environment variables for SECRET_KEY

---

**Last Updated:** April 17, 2026
**Status:** ✅ Complete and Tested
