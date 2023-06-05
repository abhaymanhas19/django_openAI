from django.contrib import admin
from apps.users.models import User,UserQuery,Files

admin.site.register(User)
admin.site.register(UserQuery)
admin.site.register(Files)

