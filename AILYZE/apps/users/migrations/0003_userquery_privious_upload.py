# Generated by Django 4.2.1 on 2023-06-01 09:57

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0002_userquery_alter_user_username'),
    ]

    operations = [
        migrations.AddField(
            model_name='userquery',
            name='privious_upload',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='privious_qution', to='users.userquery'),
        ),
    ]
