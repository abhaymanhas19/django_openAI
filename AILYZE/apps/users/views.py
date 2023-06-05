from django.shortcuts import render,HttpResponse,redirect
from apps.users.forms import UserChangePassword,RegisterUser,SummerizeType,SPecificQuestion,ThemeType,IdentifyViewpoint,CompareViewpoint,UplaodFileForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.views import LogoutView
from django.urls import reverse_lazy
from django.views import View
from apps.users.models import User,UserQuery,Files
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth import update_session_auth_hash
from apps.users.enum import Anaylsis
from django.views.generic.list import ListView
from apps.users.utils import UploadFiles



# Create your views here.


class Home(View):
    def get(self,request):
        return render(request,'index.html')
    
@method_decorator(login_required, name='dispatch')
class UserProfile(View):
    def get(self,request):
        return render(request,'userprofile.html')
    

class LoginView(View):
    def get(self,request):
        fm=AuthenticationForm()
        return render(request,'login.html',{"form":fm})
    
    def post(self, request):
        obj = AuthenticationForm(request=request, data=request.POST)
        if obj.is_valid():
            email=obj.cleaned_data['username']
            password=obj.cleaned_data['password']
            user=authenticate(email=email,password=password)
            if user is not None:
                login(request,user)
                return redirect('user-profile')
        else:
            return render(request, 'login.html', {"form": obj})
        
class LogoutView(LogoutView):
    next_page=reverse_lazy("home")

class Register(View):
    def get(self,request):
        fm=RegisterUser()
        return render(request,'register.html',{"form":fm})
    
    def post(self,request):
        fm=RegisterUser(request.POST)
        if fm.is_valid():
            fm.save()
            return redirect('login')
        else:
            return render(request,'register.html',{'form':fm})

@method_decorator(login_required, name='dispatch')
class ChangePassword(View):
    def get(self,request):
        fm=UserChangePassword(user=request.user)
        return render(request,"changepassword.html",{'form':fm})
    
    def post(self,request):
        fm=UserChangePassword(user=request.user, data=request.POST)
        if fm.is_valid():
            user=fm.save()
            update_session_auth_hash(request,user)
            return redirect('user-profile')
        return render(request,"changepassword.html",{'form':fm})
    









class UploadFileChoice(View):
    def get(self, request):
        form = UplaodFileForm()
        # files=Files.objects.all()
        context = {'form': form
                   }
        return render(request, "filedata.html", context)

    def post(self, request):
        user = self.request.user
        form = UplaodFileForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()
            obj.email = user.email
            obj.save()
            file=request.FILES.get('file')
            instance=UploadFiles(files=file)
            data=instance.upload_documents(3,10000)
            print("-----------------------------")
            print(data)
            return redirect('/get-form')
        return render(request, "filedata.html", {'form': form
                   })
    

        
            
  


class ShowFiles(View):
    def get(self, request,value):
        print(value)
        print(self.kwargs.get('upload_option'))
        value = request.GET.get('value')
        print(value)






class Getchoices(View):
    def get(self,request):
        choices=Anaylsis.choices()
        context={
            'choices':choices,
        }
        return render(request,"choices.html",context)
    


class UserQuery(View):
    a = {
        Anaylsis.Summarize.value: lambda request: render(request,'chioceform.html',{'forms':SummerizeType()}),
        Anaylsis.Ask_a_specific_question.value:  lambda request:render(request,'chioceform.html',{'forms':SPecificQuestion()}),
        Anaylsis.Conduct_thematic_analysis.value: lambda request: render(request,'chioceform.html',{'forms':ThemeType()}),
        Anaylsis.Identidy_which_document_contain_a_certain_viewpoint.value:  lambda request: render(request,'chioceform.html',{'forms':IdentifyViewpoint()}),
        Anaylsis.Compare_viewpoints_across_documents.value:  lambda request: render(request,'chioceform.html',{'forms':CompareViewpoint()})
    }

    # def get(self,request):
    #     render_fun = self.a.get(choice)
    #     if not render_fun:
    #         return render(request,'<div>Invalid choice</div>',{'forms':SummerizeType()}),
    #     return render_fun(request)
    def post(self, request):
        choice = request.POST.get('choice')
        data=request.session['choice']=choice
        render_fun = self.a.get(data)
        if not render_fun:
            return render(request,'chioceform.html',{'forms':SummerizeType()}),
        return render_fun(request)
    


class ProcessQuery(View):
    a = {
        Anaylsis.Summarize.value: lambda request: render(request,'chioceform.html',{'forms':SummerizeType()}),
        Anaylsis.Ask_a_specific_question.value:  lambda request:render(request,'chioceform.html',{'forms':SPecificQuestion()}),
        Anaylsis.Conduct_thematic_analysis.value: lambda request: render(request,'chioceform.html',{'forms':ThemeType()}),
        Anaylsis.Identidy_which_document_contain_a_certain_viewpoint.value:  lambda request: render(request,'chioceform.html',{'forms':IdentifyViewpoint()}),
        Anaylsis.Compare_viewpoints_across_documents.value:  lambda request: render(request,'chioceform.html',{'forms':CompareViewpoint()})
    }

    def post(self,request):
        choice=request.session.get('choice')
        print("-------------------------------------",choice)
        render_fun=self.a.get(choice)
        return render_fun(request)



    
