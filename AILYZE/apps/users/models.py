from typing import Any
from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils.translation import gettext_lazy as _
from apps.users.enum import Anaylsis



class User_manager(BaseUserManager):
    use_in_migrations = True

    def create_user(self, email,password,confirm_password=None, **extra_fields):
        if not email:
            raise ValueError("The email field must be required")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user


    def create_superuser(self, email, password, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)
        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True")
        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    username = models.CharField(max_length=150,unique=True)
    email = models.EmailField(unique=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    is_active = models.BooleanField(
        default=True,
    )

    objects = User_manager()


    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]


    def __str__(self):
        return self.email
    
class Files(models.Model):
    file=models.FileField(_("file"), upload_to='files', max_length=100)
    email=models.EmailField(null=True,blank=True)





class UserQuery(models.Model):
    user=models.ForeignKey(User, verbose_name=_("user_query"), on_delete=models.CASCADE,null=True,blank=True)
    question=models.TextField(_("question"))
    answer=models.TextField(_("answer"))
    date=models.DateField(_("date"), auto_now_add=True)
    time=models.TimeField(_("time"), auto_now_add=True)
    objects = models.Manager()  



