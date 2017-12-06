import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import PropTypes from 'prop-types';
import ChatBot, { Loading } from 'react-simple-chatbot';
import SYSearch from './Components/SYSearch'


class App extends Component {

  logout() {
      this.props.auth.logout();
  }

  render() {
    return (
      <div className="App" >
      <button
        className="logout_button"
        onClick={this.logout.bind(this)}
      >
        Log Out
      </button>
      <div className="bot">
      <ChatBot
    steps={[
      {
        id: '1',
        message: 'Type something to search on Syllabus. (Ex.: What is the class time)',
        trigger: 'search',
      },
      {
        id: 'search',
        user: true,
        trigger: '3',
      },
      {
        id: '3',
        component: <SYSearch />,
        waitAction: true,
        trigger: '1',
      },
    ]}
  />
  </div>
</div>
    );
  }
}

export default App;
