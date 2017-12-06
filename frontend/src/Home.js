import React, { Component } from 'react';
import { Navbar, Button, Jumbotron, NavItem, NavLink } from 'react-bootstrap';
import './App.css';

class Home extends Component {

  login() {
    this.props.auth.login();
  }

  render() {
    const { isAuthenticated } = this.props.auth;

    return (
      <div className="home-page">
        <div className="home-page__container">
        <h1 className="header"> Syllabot </h1>

      {
        !isAuthenticated() && (
            <Button

              bsStyle="primary"
              className="login_button"
              onClick={this.login.bind(this)}
            >
              Log In
            </Button>
          )
      }
      {
        !isAuthenticated() && (
            <Button
              bsStyle="primary"
              className="hmsignup_button"
              onClick={this.login.bind(this)}
            >
            Sign Up
            </Button>
          )
      }
        </div>
      </div>
    );
  }
}

export default Home;
