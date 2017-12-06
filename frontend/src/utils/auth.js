import auth0 from 'auth0-js';

import history from '../history';
//import { postUserData } from './api';

export default class AuthService {
  auth0 = new auth0.WebAuth({
    domain: 'gymbuddyios.auth0.com',
    clientID: 'aoBbzOkuq687XSK9lR2yOqKysVB6XmSJ',
    redirectUri: 'https://salty-fortress-54446.herokuapp.com/callback',
    audience: 'https://gymbuddyios.auth0.com/userinfo',
    responseType: 'token id_token',
    scope: 'openid profile'
  });

  constructor() {
    this.login = this.login.bind(this);
    this.logout = this.logout.bind(this);
    this.handleAuthentication = this.handleAuthentication.bind(this);
    this.isAuthenticated = this.isAuthenticated.bind(this);
  }


  login() {
    this.auth0.authorize();
  }
  handleAuthentication() {
      this.auth0.parseHash((err, authResult) => {
        if (authResult && authResult.accessToken && authResult.idToken) {
          this.getUserInfo(authResult);
          this.setSession(authResult);
          history.replace('/bot');
        } else if (err) {
          //history.replace('/courses');
          console.log(err);
        }
      });


    }

    setSession(authResult) {
      // Set the time that the access token will expire at
      let expiresAt = JSON.stringify((authResult.expiresIn * 1000) + new Date().getTime());
      localStorage.setItem('access_token', authResult.accessToken);
      localStorage.setItem('id_token', authResult.idToken);
      localStorage.setItem('expires_at', expiresAt);
      // navigate to the home route
    //  history.replace('/courses');
    }

    getUserInfo(authResult){

    this.auth0.client.userInfo(authResult.accessToken, (err, profile) => {
      if(profile) {
          var userid = profile.sub.split('|');
          localStorage.setItem('uid', userid[1]);
          var name = profile.nickname;
        //  postUserData(userid[1],name);
      }
      if(err){
        console.log('rekt');
      }
    });
}
    logout() {
      // Clear access token and ID token from local storage
      localStorage.removeItem('access_token');
      localStorage.removeItem('id_token');
      localStorage.removeItem('expires_at');
      // navigate to the home route
      history.replace('/');
    }

    isAuthenticated() {
      // Check whether the current time is past the
      // access token's expiry time
      let expiresAt = JSON.parse(localStorage.getItem('expires_at'));
      return new Date().getTime() < expiresAt;
    }
    requireAuth() {
      if (!this.isAuthenticated()) {
        history.replace('/');
      }
}
}
