#!/usr/bin/env python3
"""
User management system for persistent storage of user credentials.
"""

import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

class UserManager:
    def __init__(self, users_file='users.json'):
        self.users_file = users_file
        self.users = self.load_users()
    
    def load_users(self):
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def save_users(self):
        """Save users to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving users: {e}")
            return False
    
    def add_user(self, email, password, name):
        """Add a new user"""
        if email in self.users:
            return False, "User already exists"
        
        self.users[email] = {
            'password': generate_password_hash(password),
            'name': name
        }
        
        if self.save_users():
            return True, "User created successfully"
        else:
            return False, "Error saving user"
    
    def verify_user(self, email, password):
        """Verify user credentials"""
        if email not in self.users:
            return False, None
        
        user = self.users[email]
        if check_password_hash(user['password'], password):
            return True, user['name']
        
        return False, None
    
    def get_user(self, email):
        """Get user information"""
        return self.users.get(email)
    
    def update_user(self, email, **kwargs):
        """Update user information"""
        if email not in self.users:
            return False, "User not found"
        
        for key, value in kwargs.items():
            if key == 'password':
                self.users[email][key] = generate_password_hash(value)
            else:
                self.users[email][key] = value
        
        if self.save_users():
            return True, "User updated successfully"
        else:
            return False, "Error saving user"
    
    def delete_user(self, email):
        """Delete a user"""
        if email not in self.users:
            return False, "User not found"
        
        del self.users[email]
        
        if self.save_users():
            return True, "User deleted successfully"
        else:
            return False, "Error saving changes"
    
    def get_all_users(self):
        """Get all users (for admin purposes)"""
        return {email: {'name': user['name']} for email, user in self.users.items()}
    
    def create_default_admin(self):
        """Create a default admin user if no users exist"""
        if not self.users:
            success, message = self.add_user(
                email="admin@gmail.com",
                password="Admin123!",
                name="Administrator"
            )
            if success:
                print("Default admin user created:")
                print("Email: admin@gmail.com")
                print("Password: Admin123!")
            return success, message
        return True, "Users already exist"

# Global user manager instance
user_manager = UserManager() 