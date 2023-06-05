from django.contrib.auth.forms import PasswordChangeForm,UserCreationForm,AuthenticationForm
from apps.users.models import User,Files
from django import forms
from django.utils.translation import gettext_lazy as _
from apps.users.enum import Anaylsis
from multiupload.fields import MultiFileField

class RegisterUser(UserCreationForm):
    class Meta:
        model=User
        fields=['username','email','password1','password2']


class UserChangePassword(PasswordChangeForm):
    class Meta:
        model=User
        fields=['old_password','new_password1','new_password2']



class UplaodFileForm(forms.ModelForm):
    UPLOAD_CHOICES = [
        ('new', 'Upload New File'),
        ('previous', 'Upload Files from Previous Uploads'),
    ]
    upload_option = forms.ChoiceField(choices=UPLOAD_CHOICES)
    # file=MultiFileField(min_num=1)
    email = forms.EmailField(required=False)
    class Meta:
        model = Files
        fields = ['upload_option', 'file', 'email']


    
    # def clean_file(self):
    #     file = self.cleaned_data.get('file')
    #     existing_file = Files.objects.filter(file=file).first()
    #     if existing_file:
    #         return existing_file.file

    #     return file


    # def get_initial(self):
    #     initial = super(UplaodFileForm, self).get_initial()
    #     initial['email'] = self.request.user.email
    #     return initial



            








class SummerizeType(forms.Form):
    SUMMARY_CHOICES = [
        ('Essay', 'Essay'),
        ('Bullet points', 'Bullet points'),
    ]
    summary=forms.ChoiceField(choices=SUMMARY_CHOICES)
    instruction=forms.CharField(widget=forms.Textarea(
        attrs={'placeholder':'Additional instructions regarding your desired response from chatbot (e.g., length, style)'}
    ))
    
class SPecificQuestion(forms.Form):
    question=forms.CharField(error_messages={
        'required': 'Please enter your question option.'},widget=forms.Textarea(
        attrs={'placeholder':'Enter your Question'}
    ))
    instruction=forms.CharField(widget=forms.Textarea(
        attrs={'placeholder':'Additional instructions regarding your desired response from chatbot (e.g., length, style)'}
    ))
    keywords=forms.CharField(widget=forms.Textarea(
        attrs={'placeholder':'Keywords or section names to help chatbot extract relevant parts of the document to analyze'}
    ))


class ThemeType(forms.Form):
    SUMMARY_CHOICE = [
        ('Codebook', 'Codebook'),
        ('Essay', 'Essay'),
    ]
    theme_type=forms.ChoiceField(choices=SUMMARY_CHOICE)
    instruction=forms.CharField(widget=forms.Textarea(
        attrs={'placeholder':'Additional instructions regarding your desired response'}
    ))


class IdentifyViewpoint(forms.Form):
    instruction=forms.CharField(widget=forms.Textarea(
        attrs={'placeholder':'Participants generally trust medical professionals about the flu and vaccine'}
    ))

class CompareViewpoint(forms.Form):
    question=forms.CharField(error_messages={
        'required': 'Please enter your question option.'},widget=forms.Textarea(
        attrs={'placeholder':'Input your Question'}
    ))
    instruction=forms.CharField(widget=forms.Textarea(
        attrs={'placeholder':'Participant 1, 2, 3, 4, 5 and 6 are male. Participants 7, 8, 9, 10, 11 and 12 are female."'}
    ))


    
    
