import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import PropTypes from 'prop-types';
import ChatBot, { Loading } from 'react-simple-chatbot';
import SYSearch from './Components/SYSearch'


class App extends Component {
  render() {
    return (
      <div className="App" >
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
    );
  }
}

export default App;
