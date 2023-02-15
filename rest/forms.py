#from django import forms
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm

class LoginForm(AuthenticationForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['placeholder'] = field.label#全てのフォームの部品にplaceholderを定義して、入力フォームにフォーム名が表示されるように指定する
#class TextForm(forms.Form):
#    jtext = forms.CharField(
#        label='出題文',
#        max_length=200,
#        required=True,
#    )
#    title = forms.CharField(
#        label='学習者訳',
#        max_length=200,
#        required=True,
#    )


class SampleChoiceForm(forms.Form):
	tensaku_score = (
		('1', '1:よくない'),
		('2', '2:少しよい'),
        ('3', '3:よい'),
        ('4', '4:とてもよい'),
	)
	select = forms.fields.ChoiceField(required=True, widget=forms.Select(attrs={'class': 'custom-select my-1 mr-sm-2', 'id': 'inlineFormCustomSelectPref'}), choices=tensaku_score)

class FinalChoiceForm1(forms.Form):
	final_score1 = (
		('1', '1:よくない'),
		('2', '2:少しよい'),
        ('3', '3:よい'),
        ('4', '4:とてもよい'),
	)
	final_select1 = forms.fields.ChoiceField(required=True, widget=forms.Select(attrs={'class': 'custom-select my-1 mr-sm-2', 'id': 'inlineFormCustomSelectPref'}), choices=final_score1)

class FinalChoiceForm2(forms.Form):
	final_score2 = (
		('1', '1:全く役に立たない'),
		('2', '2:少し役に立つ'),
        ('3', '3:役に立つ'),
        ('4', '4:とても役に立つ'),
	)
	final_select2 = forms.fields.ChoiceField(required=True, widget=forms.Select(attrs={'class': 'custom-select my-1 mr-sm-2', 'id': 'inlineFormCustomSelectPref'}), choices=final_score2)